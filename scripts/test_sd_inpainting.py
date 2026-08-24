import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Test SD Inpainting on an image")
    parser.add_argument("--image", type=str, default="outputs/immunized_image.png", help="Path to the input immunized image")
    parser.add_argument("--mask", type=str, default="data/train/masks/mask_00000.png", help="Path to the mask image")
    parser.add_argument("--prompt", type=str, default="a person wearing a red jacket", help="Prompt for inpainting")
    parser.add_argument("--output", type=str, default="outputs/sd_edited_result.png", help="Path to save the generated result")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    
    # 1. Load Images
    if not os.path.exists(args.image):
        print(f"Error: Could not find input image at {args.image}.")
        print("Please run test_unet.py first to generate the immunized_image.png")
        return
        
    input_pil = Image.open(args.image).convert("RGB").resize((512, 512))
        
    if not os.path.exists(args.mask):
        print(f"Warning: {args.mask} not found. Creating a dummy mask for testing.")
        mask_pil = Image.new("L", (512, 512), color="white")
    else:
        mask_pil = Image.open(args.mask).convert("L").resize((512, 512))
        
    # 2. Load Pipeline
    print("Loading SD Inpainting Pipeline (FP16)...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16
    ).to(device)
    # pipe.enable_xformers_memory_efficient_attention() # Optional to save VRAM
    
    # 3. Inference
    print(f"Running SD Inpainting with prompt: '{args.prompt}'")
    print("This will perform 50 diffusion steps...")
    
    result = pipe(
        prompt=args.prompt,
        image=input_pil,
        mask_image=mask_pil,
        num_inference_steps=50
    ).images[0]
    
    # 4. Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result.save(args.output)
    print(f"\nSuccess! Saved result to {args.output}")

if __name__ == "__main__":
    main()
