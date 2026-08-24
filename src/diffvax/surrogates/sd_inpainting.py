import torch
from diffusers import StableDiffusionInpaintPipeline
from diffvax.surrogates.base import BaseSurrogate
import torch.nn.functional as F

class SDInpaintingSurrogate(BaseSurrogate):
    """
    Model A: Stable Diffusion v1.5 Inpainting.
    Strategy: Early-timestep attack (inspired by DiffusionGuard).
    Loss: Maximize the L2 norm of the predicted noise at the initial timestep T.
    """
    def __init__(self, model_id="runwayml/stable-diffusion-inpainting", timestep=50):
        super().__init__()
        # Load the pipeline in fp16 to save memory
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        self.timestep = timestep
        
        # We only need the individual components
        self.unet = self.pipe.unet
        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        self.scheduler = self.pipe.scheduler
        
        # Freeze all components to avoid OOM and unintended training
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        # Memory optimization for Colab (xformers)
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            print("xformers not available. Continuing without memory efficient attention.")

    def _get_text_embeddings(self, prompts, device):
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input_ids)[0]
            
        return text_embeddings

    def compute_loss(self, immunized_images, original_images, masks, prompts):
        """
        Computes the early-timestep attack loss for SD Inpainting.
        
        Args:
            immunized_images: (B, 3, H, W) range [-1, 1], output of immunizer U-Net.
            original_images: (B, 3, H, W) not strictly needed here unless we use it for masked area.
            masks: (B, 1, H, W) where 1 is the region to be inpainted.
            prompts: List[str] length B.
        """
        device = immunized_images.device
        dtype = self.unet.dtype
        batch_size = immunized_images.shape[0]
        
        # Ensure inputs match the model's precision
        immunized_images = immunized_images.to(dtype)
        original_images = original_images.to(dtype)
        masks = masks.to(dtype)
        
        # 1. Get Text Embeddings
        text_embeddings = self._get_text_embeddings(prompts, device).to(dtype)
        
        # 2. Encode Immunized Image to Latents
        # We must NOT use no_grad here because the gradients need to flow back to immunized_images
        latents = self.vae.encode(immunized_images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        
        # 3. Add noise at the target early timestep T
        noise = torch.randn_like(latents)
        t = torch.tensor([self.timestep] * batch_size, device=device, dtype=torch.long)
        noisy_latents = self.scheduler.add_noise(latents, noise, t)
        
        # 4. Prepare Mask and Masked Image
        # Masked image is the part of the image we KEEP. 
        # Usually masks == 1 is the inpainted region, so we keep region where mask == 0.
        masked_images = immunized_images * (1 - masks)
        masked_image_latents = self.vae.encode(masked_images).latent_dist.sample()
        masked_image_latents = masked_image_latents * self.vae.config.scaling_factor
        
        # Downsample mask to latent resolution
        mask_latents = F.interpolate(masks, size=latents.shape[-2:], mode="nearest")
        
        # 5. Concatenate inputs for Inpainting UNet
        # input structure: [noisy_latents, mask, masked_image_latents]
        latent_model_input = torch.cat([noisy_latents, mask_latents, masked_image_latents], dim=1)
        
        # 6. Predict Noise
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings
        ).sample
        
        # 7. Compute Loss
        # Objective: Maximize the L2 norm of the predicted noise
        # Since the optimizer minimizes, we return the negative L2 norm.
        # Alternatively, we could maximize MSE against the added noise.
        # Following the spec: "Maximizar a norma L2 do ruído predito"
        l2_norm = torch.norm(noise_pred, p=2, dim=(1,2,3)).mean()
        loss = -l2_norm
        
        return loss
