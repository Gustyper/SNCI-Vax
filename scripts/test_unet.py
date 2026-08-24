import torch
from PIL import Image
import os
import sys

# Ensure src is in python path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_project_root, "src"))

from diffvax.model import NestedUNet
from diffvax.utils import prepare_image_return_3d

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    
    # 1. Load U-Net (Immunizer)
    # Uncomment the line below to load your trained weights
    unet = NestedUNet(num_classes=3, input_channels=3).to(device)
    # unet.load_state_dict(torch.load("path/to/diffvax_epoch_5.pth", map_location=device))
    unet.eval()
    
    # 2. Load Image
    img_path = "data/train/images/00000.png"
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} not found. Creating a dummy blue image for testing.")
        original_pil = Image.new("RGB", (512, 512), color="blue")
    else:
        original_pil = Image.open(img_path).convert("RGB").resize((512, 512))
    
    # Prepare tensor [-1, 1] for U-Net
    img_tensor = prepare_image_return_3d(original_pil).unsqueeze(0).to(device)
    
    # 3. Forward Pass (Immunization)
    print("Generating adversarial noise via U-Net...")
    with torch.no_grad():
        noise = unet(img_tensor)
        alpha = 4 / 255.0
        # Clamp noise to ensure imperceptibility
        noise = torch.clamp(noise, -alpha, alpha)
        # Add noise to the ENTIRE image
        immunized_tensor = torch.clamp(img_tensor + noise, -1.0, 1.0)
        
    # Convert tensor back to PIL Image
    immunized_tensor_norm = (immunized_tensor / 2 + 0.5).clamp(0, 1)
    immunized_np = immunized_tensor_norm.cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
    immunized_pil = Image.fromarray((immunized_np * 255).astype("uint8"))
    
    # 4. Save results
    os.makedirs("outputs", exist_ok=True)
    immunized_pil.save("outputs/immunized_image.png")
    print("Saved immunized image to outputs/immunized_image.png")

if __name__ == "__main__":
    main()
