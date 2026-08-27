#!/usr/bin/env python3
"""Train DiffVax immunization model using Accelerate."""

import argparse
import os
import sys
import yaml
import csv
from pathlib import Path
from tqdm import tqdm

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_project_root, "src"))

from diffvax.model import NestedUNet
from diffvax.surrogates.sd_inpainting import SDInpaintingSurrogate
from diffvax.dataset.diffvax_dataset import DiffVaxDataset
from diffvax.utils import get_train_val_image_prompt_list, ensure_dataset_in_data_dir

def main():
    parser = argparse.ArgumentParser(description="Train DiffVax Ensemble Attack")
    parser.add_argument("--config", type=str, default="configs/train.yml")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--disable-sd15", action="store_true", help="Disable SD 1.5 Surrogate")
    parser.add_argument("--masks-subdir", type=str, default=None, help="Override masks directory (e.g. train/maks_clothing)")
    parser.add_argument("--mask-prefix", type=str, default=None, help="Prefix for mask files (e.g. mask_clothing_)")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    config["data_dir"] = args.data_dir
    config["output_dir"] = args.output_dir
    
    os.makedirs(args.output_dir, exist_ok=True)
    history_csv = os.path.join(args.output_dir, "loss_history.csv")
    
    # 1. Initialize Accelerator
    active_surrogates = []
    if not args.disable_sd15:
        active_surrogates.append("sd15")
        
    num_surrogates = len(active_surrogates) if len(active_surrogates) > 0 else 1
    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=num_surrogates)

    # 2. Models & Optimizer
    # Immunizer
    immunizer_unet = NestedUNet(num_classes=3, input_channels=3).train()
    optimizer = torch.optim.AdamW(immunizer_unet.parameters(), lr=float(config["learning_rate"]))

    # Surrogates
    surrogate_sd = None
    if "sd15" in active_surrogates:
        accelerator.print("Loading SD 1.5 Inpainting Surrogate...")
        surrogate_sd = SDInpaintingSurrogate().eval()
        # Ensure it's frozen
        for param in surrogate_sd.parameters():
            param.requires_grad = False

    # 3. Dataset & DataLoader
    accelerator.print("Preparing Datasets...")
    data_dir = ensure_dataset_in_data_dir("ozdentarikcan/DiffVaxDataset", data_dir=args.data_dir)
    train_list, val_list = get_train_val_image_prompt_list(data_dir)
    
    train_dataset = DiffVaxDataset(
        train_list, 
        data_dir, 
        images_subdir=config.get("images_subdir", "train/images"),
        masks_subdir=args.masks_subdir if args.masks_subdir else config.get("masks_subdir", "train/masks"),
        mask_prefix=args.mask_prefix if args.mask_prefix else config.get("mask_prefix", "mask_")
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    
    # 4. Prepare everything with Accelerator
    immunizer_unet, optimizer, train_loader = accelerator.prepare(immunizer_unet, optimizer, train_loader)
    
    if surrogate_sd:
        surrogate_sd.to(accelerator.device)

    # 5. Training Loop
    iter_num = config.get("iter_num", 10000)
    
    if accelerator.is_main_process:
        with open(history_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "total_loss", "sd15_loss"])

    global_step = 0
    epoch = 0
    
    progress_bar = tqdm(total=iter_num, disable=not accelerator.is_local_main_process, desc="Training")
    
    while global_step < iter_num:
        for batch in train_loader:
            if global_step >= iter_num:
                break
                
            original_images = batch["image"]
            masks = batch["mask"]
            prompts = batch["prompt"]
            
            with accelerator.accumulate(immunizer_unet):
                # Forward pass of immunizer
                noise = immunizer_unet(original_images) 
                alpha = config["alpha"] / 255.0
                noise = torch.clamp(noise, -alpha, alpha)
                immunized_images = torch.clamp(original_images + noise, -1.0, 1.0)
                
                total_loss = 0.0
                sd15_loss_val = 0.0
                
                # --- Surrogate A: SD Inpainting ---
                if surrogate_sd:
                    sd_loss = surrogate_sd.compute_loss(immunized_images, original_images, masks, prompts)
                    total_loss += sd_loss
                    sd15_loss_val = sd_loss.item()
                    torch.cuda.empty_cache()
                
                accelerator.backward(total_loss)
                optimizer.step()
                optimizer.zero_grad()
                
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{total_loss.item():.4f}"})
            
            if accelerator.is_main_process:
                with open(history_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([global_step, total_loss.item(), sd15_loss_val])
                    
            global_step += 1
            
        epoch += 1
        
        # Save checkpoint periodically
        if accelerator.is_main_process and epoch % 5 == 0:
            save_path = os.path.join(args.output_dir, f"diffvax_epoch_{epoch}.pth")
            unwrapped_model = accelerator.unwrap_model(immunizer_unet)
            torch.save(unwrapped_model.state_dict(), save_path)

    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "diffvax_final.pth")
        unwrapped_model = accelerator.unwrap_model(immunizer_unet)
        torch.save(unwrapped_model.state_dict(), save_path)
        accelerator.print(f"Training Complete! Final model saved to {save_path}")

if __name__ == "__main__":
    main()
