import torch
import torch.nn as nn
import torch.nn.functional as F


class CircularOctaveLoss(nn.Module):
    def __init__(self, num_classes=12 * 3, reduction="mean"):
        super(CircularOctaveLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def circular_distance(self, pred_class, target_class):
        dist = torch.abs(pred_class - target_class) % self.num_classes
        return torch.min(dist, self.num_classes - dist)

    def forward(self, pred, target):
        # Find the predicted pitch class and octave shift
        pred_class = pred % self.num_classes
        pred_octave = pred // self.num_classes

        # Ground truth pitch class and octave
        target_class = target % self.num_classes
        target_octave = target // self.num_classes

        # Circular distance for pitch class within an octave
        class_distance = self.circular_distance(pred_class, target_class)

        # Octave penalty
        octave_distance = torch.abs(pred_octave - target_octave)

        # Total loss = class distance + octave penalty
        loss = class_distance + octave_distance

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()


class ZeroOneLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(ZeroOneLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        loss = (pred != target).float()
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()


class OctaveZeroOneLoss(nn.Module):
    def __init__(self, num_classes=12 * 3, reduction="mean"):
        super(OctaveZeroOneLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, pred, target):
        pred_class = pred % self.num_classes
        pred_octave = pred // self.num_classes

        target_class = target % self.num_classes
        target_octave = target // self.num_classes

        class_loss = (pred_class != target_class).float()
        octave_loss = 0.5 * (pred_octave != target_octave).float()

        loss = class_loss + octave_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()


class LabelCrossEntropy(nn.Module):
    def __init__(
        self, num_classes: int, mode: str = "distance", reduction: str = "mean"
    ):
        super(LabelCrossEntropy, self).__init__()
        assert mode in ["distance", "absolute", "octave"]
        self.mode = mode
        self.num_classes = num_classes
        self.reduction = reduction

        self.distance_loss = CircularOctaveLoss(num_classes=12 * 3, reduction=reduction)
        self.zero_one_loss = ZeroOneLoss(reduction=reduction)
        self.octave_zero_one_loss = OctaveZeroOneLoss(
            num_classes=12 * 3, reduction=reduction
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Computes the cross-entropy loss between the predicted and target labels
            subject to some musically motivated constraints.
        Args:
            pred (torch.Tensor): the predicted labels, shape (batch_size)
            target (torch.Tensor): the target labels, shape (batch_size)

        Returns:
            torch.Tensor: the loss value
        """
        if self.mode == "distance":
            return self.distance_loss(pred, target)
        elif self.mode == "absolute":
            return self.zero_one_loss(pred, target)
        elif self.mode == "octave":
            return self.octave_zero_one_loss(pred, target)
