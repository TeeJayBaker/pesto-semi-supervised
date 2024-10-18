import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.entropy import CrossEntropyLoss


def gaussian_peak_tensor(centre: torch.Tensor, spread: int, bins_per_semitone: int):
    """
    Generates a Gaussian peak tensor based on a center and spread.

    Args:
        centre (torch.Tensor): A tensor of shape (batch_size) indicating the center positions for each batch.
        spread (int): The standard deviation (spread) of the Gaussian distribution.
        bins_per_semitone (int): Number of bins per semitone, controls the granularity of the pitch encoding.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, 128 * bins_per_semitone) where each row is a Gaussian peak centered at the corresponding value in the 'centre' tensor.
    """
    x = (
        torch.arange(0, 128 * bins_per_semitone)
        .float()
        .unsqueeze(0)
        .expand(centre.shape[0], -1)
    ).to(centre.device)
    centre = centre.unsqueeze(1)
    return torch.exp(-((x - centre) ** 2) / (2 * spread**2))


def gaussian_cosine_tensor(centre: torch.Tensor, spread: int, bins_per_semitone: int):
    """
    Generates a tensor combining a Gaussian distribution with a cosine modulation peaking each octave.

    Args:
        centre (torch.Tensor): A tensor of shape (batch_size) indicating the center positions for each batch.
        spread (int): The standard deviation (spread) of the Gaussian distribution.
        bins_per_semitone (int): Number of bins per semitone, controls the granularity of the pitch encoding.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, 128 * bins_per_semitone) where each row is a Gaussian-scaled cosine function centered at the corresponding value in the 'centre' tensor.
    """
    x = (
        torch.arange(0, 128 * bins_per_semitone)
        .float()
        .unsqueeze(0)
        .expand(centre.shape[0], -1)
    ).to(centre.device)
    centre = centre.unsqueeze(1)
    return (
        torch.exp(-((x - centre) ** 2) / (2 * spread**2))
        * 0.5
        * (torch.cos((x - centre) * 2 * torch.pi / (12 * bins_per_semitone)) + 1)
    )


class LabelCrossEntropy(nn.Module):
    def __init__(
        self,
        bins_per_semitone: int,
        mode: str = "absolute",
        criterion: nn.Module = CrossEntropyLoss(),
    ):
        """
        Args:
            bins_per_semitone (int): Number of bins per semitone.
            mode (str): 'absolute' or 'octave' mode. Default is 'absolute'.
            criterion (nn.Module): Loss function, default is CrossEntropyLoss.
        """
        super(LabelCrossEntropy, self).__init__()
        assert mode in ["absolute", "octave"]
        self.mode = mode
        self.criterion = criterion
        self.bps = bins_per_semitone

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred (torch.Tensor): Predicted probabilities (batch_size, num_classes).
            target (torch.Tensor): Target labels (batch_size).

        Returns:
            torch.Tensor: Loss value.
        """

        labeled = ~torch.isnan(target)
        if labeled.sum() == 0:
            return pred.sum() * 0.0

        if self.mode == "absolute":
            return self.criterion(
                pred[labeled],
                gaussian_peak_tensor(
                    target[labeled] * self.bps, 0.2 * self.bps, self.bps
                ),
            )
        elif self.mode == "octave":
            return self.criterion(
                pred[labeled],
                gaussian_cosine_tensor(
                    target[labeled] * self.bps, 2 * self.bps, self.bps
                ),
            )
