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


class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving."""
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss, model_state, path):
        # We minimize loss (more negative loss means higher adversary noise norm)
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model_state, path)
            return True, False  # is_best, should_stop
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False, self.early_stop


def main():
    parser = argparse.ArgumentParser(description="Train DiffVax Ensemble Attack")
    parser.add_argument("--config", type=str, default="configs/train.yml")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--disable-sd15", action="store_true", help="Disable SD 1.5 Surrogate")
    parser.add_argument("--masks-subdir", type=str, default=None, help="Override train masks directory (e.g. train/maks_clothing)")
    parser.add_argument("--val-masks-subdir", type=str, default=None, help="Override val masks directory (e.g. validation/maks_clothing)")
    parser.add_argument("--mask-prefix", type=str, default=None, help="Prefix for mask files (e.g. mask_clothing_)")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (epochs)")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Early stopping min delta")
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
    log_with = "wandb" if args.use_wandb else None
    accelerator = Accelerator(
        mixed_precision="fp16", 
        gradient_accumulation_steps=num_surrogates,
        log_with=log_with
    )

    if args.use_wandb and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config.get("project_name", "diffvax"),
            config={**config, **vars(args)}
        )

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
    
    train_masks_subdir = args.masks_subdir if args.masks_subdir else config.get("masks_subdir", "train/masks")
    val_masks_subdir = args.val_masks_subdir if args.val_masks_subdir else (
        args.masks_subdir.replace("train", "validation") if (args.masks_subdir and "train" in args.masks_subdir) else "validation/masks"
    )
    mask_prefix = args.mask_prefix if args.mask_prefix else config.get("mask_prefix", "mask_")

    train_dataset = DiffVaxDataset(
        train_list, 
        data_dir, 
        images_subdir=config.get("images_subdir", "train/images"),
        masks_subdir=train_masks_subdir,
        mask_prefix=mask_prefix
    )
    
    val_dataset = DiffVaxDataset(
        val_list,
        data_dir,
        images_subdir=config.get("val_images_subdir", "validation/images"),
        masks_subdir=val_masks_subdir,
        mask_prefix=mask_prefix
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # 4. Prepare everything with Accelerator
    immunizer_unet, optimizer, train_loader, val_loader = accelerator.prepare(
        immunizer_unet, optimizer, train_loader, val_loader
    )
    
    if surrogate_sd:
        surrogate_sd.to(accelerator.device)

    # 5. Early Stopping setup
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    best_model_path = os.path.join(args.output_dir, "diffvax_best.pth")

    # 6. Training Loop
    iter_num = config.get("iter_num", 10000)
    
    if accelerator.is_main_process:
        with open(history_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "epoch", "total_loss", "sd15_loss", "val_loss"])

    global_step = 0
    epoch = 0
    
    progress_bar = tqdm(total=iter_num, disable=not accelerator.is_local_main_process, desc="Training")
    
    while global_step < iter_num:
        immunizer_unet.train()
        epoch_loss_sum = 0.0
        epoch_steps = 0

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
                
            epoch_loss_sum += total_loss.item()
            epoch_steps += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{total_loss.item():.4f}", "epoch": epoch})
            
            if args.use_wandb:
                accelerator.log({"train/step_loss": total_loss.item(), "train/sd15_loss": sd15_loss_val}, step=global_step)

            global_step += 1
            
        epoch += 1
        avg_train_loss = epoch_loss_sum / max(1, epoch_steps)

        # --- Validation Loop at end of each epoch ---
        immunizer_unet.eval()
        val_loss_sum = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for val_batch in val_loader:
                v_orig_images = val_batch["image"]
                v_masks = val_batch["mask"]
                v_prompts = val_batch["prompt"]
                
                v_noise = immunizer_unet(v_orig_images)
                alpha = config["alpha"] / 255.0
                v_noise = torch.clamp(v_noise, -alpha, alpha)
                v_immunized = torch.clamp(v_orig_images + v_noise, -1.0, 1.0)
                
                if surrogate_sd:
                    v_loss = surrogate_sd.compute_loss(v_immunized, v_orig_images, v_masks, v_prompts)
                    val_loss_sum += v_loss.item()
                    val_steps += 1
                    
        avg_val_loss = val_loss_sum / max(1, val_steps)
        
        if accelerator.is_main_process:
            with open(history_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([global_step, epoch, avg_train_loss, avg_train_loss, avg_val_loss])

            if args.use_wandb:
                accelerator.log({
                    "epoch": epoch,
                    "train/epoch_loss": avg_train_loss,
                    "val/loss": avg_val_loss
                }, step=global_step)

            unwrapped_model = accelerator.unwrap_model(immunizer_unet)
            is_best, should_stop = early_stopping(avg_val_loss, unwrapped_model.state_dict(), best_model_path)
            
            if is_best:
                accelerator.print(f"\n[Época {epoch}] 🔥 Novo melhor modelo salvo em {best_model_path} (val_loss: {avg_val_loss:.4f})")
            else:
                accelerator.print(f"\n[Época {epoch}] val_loss: {avg_val_loss:.4f} (Paciência: {early_stopping.counter}/{early_stopping.patience})")

            # Periodic checkpoint every 5 epochs
            if epoch % 5 == 0:
                epoch_path = os.path.join(args.output_dir, f"diffvax_epoch_{epoch}.pth")
                torch.save(unwrapped_model.state_dict(), epoch_path)

            if should_stop:
                accelerator.print(f"\n🛑 Early Stopping ativado na época {epoch}! O modelo parou de evoluir no conjunto de validação.")
                break

    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "diffvax_final.pth")
        unwrapped_model = accelerator.unwrap_model(immunizer_unet)
        torch.save(unwrapped_model.state_dict(), save_path)
        accelerator.print(f"Treinamento finalizado! Modelo final salvo em {save_path}")
        
    if args.use_wandb:
        accelerator.end_training()


if __name__ == "__main__":
    main()

