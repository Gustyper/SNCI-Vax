import torch
from torch.utils.data import Dataset
from diffvax.utils import load_image, prepare_mask_and_masked_image

class DiffVaxDataset(Dataset):
    def __init__(self, image_prompt_list, data_dir, images_subdir="train/images", masks_subdir="train/masks"):
        """
        Args:
            image_prompt_list (list): List of dicts with 'image' (filename) and 'prompts' (list of strings).
            data_dir (str): Base data directory.
        """
        self.data_dir = data_dir
        self.images_subdir = images_subdir
        self.masks_subdir = masks_subdir
        
        # Flatten prompts so each sample has exactly one prompt
        self.samples = []
        for item in image_prompt_list:
            image_name = item["image"][:-4] if item["image"].endswith(".png") else item["image"]
            for prompt in item["prompts"]:
                self.samples.append({
                    "image_name": image_name,
                    "prompt": prompt
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_name = sample["image_name"]
        prompt = sample["prompt"]

        image = load_image(
            image_name,
            self.data_dir,
            is_mask=False,
            images_subdir=self.images_subdir,
            masks_subdir=self.masks_subdir,
        )
        image_mask = load_image(
            image_name,
            self.data_dir,
            is_mask=True,
            images_subdir=self.images_subdir,
            masks_subdir=self.masks_subdir,
        )
        
        mask_torch, image_torch, non_masked_image_torch = prepare_mask_and_masked_image(
            image, image_mask
        )
        
        return {
            "image": image_torch.squeeze(0), # (3, H, W)
            "mask": mask_torch.squeeze(0),   # (1, H, W)
            "prompt": prompt
        }
