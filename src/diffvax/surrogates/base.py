import torch
import torch.nn as nn

class BaseSurrogate(nn.Module):
    """
    Base class for all surrogate models in the ensemble attack.
    All surrogates must be frozen (requires_grad=False) and implement `compute_loss`.
    """
    def __init__(self):
        super().__init__()

    def compute_loss(self, immunized_images, original_images, **kwargs):
        """
        Computes the adversarial loss to be backpropagated to the immunizer U-Net.
        
        Args:
            immunized_images (torch.Tensor): The output of the U-Net.
            original_images (torch.Tensor): The original pristine images.
            **kwargs: Extra arguments like 'prompts' or 'masks' depending on the model.
            
        Returns:
            torch.Tensor: A scalar loss value.
        """
        raise NotImplementedError("Subclasses must implement compute_loss()")
