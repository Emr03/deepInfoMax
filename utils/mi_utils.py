import torch

class Permute(torch.nn.Module):
    """Module for permuting axes.
    """
    def __init__(self, *perm):
        """
        Args:
            *perm: Permute axes.
        """
        super().__init__()
        self.perm = perm

    def forward(self, input):
        """Permutes axes of tensor.
        Args:
            input: Input tensor.
        Returns:
            torch.Tensor: permuted tensor.
        """
        return input.permute(*self.perm)
