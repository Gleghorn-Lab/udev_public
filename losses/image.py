import torch
import torch.nn.functional as F
import torch.nn as nn
from dtw import dtw


# Losses
# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    """
    Forward pass of DiceBCELoss
    | Params:
    |  - inputs: Predicted segmentation tensor
    |  - targets: Ground truth segmentation tensor
    |  - smooth: Smoothing factor
    | Returns:
    |  - Dice_BCE: combined loss value
    """

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.flatten()
        targets = targets.flatten()

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class MultiNoiseLoss(nn.Module):
    def __init__(self, n_losses):
        """
        Initialise the module, and the scalar "noise" parameters (sigmas in arxiv.org/abs/1705.07115).
        If using CUDA, requires manually setting them on the device, even if the model is already set to device.
        """
        super(MultiNoiseLoss, self).__init__()
        self.n_losses = n_losses
        if torch.cuda.is_available():
            self.noise_params = torch.rand(n_losses, requires_grad=True, device="cuda:0")
        else:
            self.noise_params = torch.rand(n_losses, requires_grad=True)

    """
    DiceBCELoss function
    | Params:
    |  - inputs: Predicted segmentation tensor
    |  - targets: Ground truth segmentation tensor
    |  - smooth: Smoothing factor
    | Returns:
    |  - Dice_BCE: combined loss value
    """

    def DiceBCELoss(self, inputs, targets, smooth=1):
        inputs = inputs.flatten()
        targets = targets.flatten()
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy_with_logits(inputs, targets)
        Dice_BCE = BCE + dice_loss
        return Dice_BCE

    """
    Forward pass of MultiNoiseLoss module
    | Computes the total loss as a function of a list of classification losses
    | Params:
    |  - SR: Predicted segmentation tensor
    |  - GT: Ground Truth segmentation tensor
    | Returns:
    |  - total_loss: Total loss value
    """

    def forward(self, SR, GT):
        """
        Computes the total loss as a function of a list of classification losses.
        TODO: Handle regressions losses, which require a factor of 2 (see arxiv.org/abs/1705.07115 page 4)
        """

        losses = [self.DiceBCELoss(SR[:, :, :, i], GT[:, :, :, i]) for i in range(self.n_losses)]

        total_loss = 0
        for i, loss in enumerate(losses):
            total_loss += (1 / torch.square(self.noise_params[i])) * loss + torch.log(self.noise_params[i])

        return total_loss


class DiceIOULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceIOULoss, self).__init__()

    """
    Forward pass of the DiceIOULoss module
    | Params:
    |  - inputs: Predicted segmentation tensor
    |  - targets: Ground truth segmentation tensor
    |  - smooth: Smoothing factor
    | Returns:
    |  - Dice_IOU: Combined loss value
    """

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.flatten()
        targets = targets.flatten()

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        IOU_loss = 1 - (intersection + smooth) / (union + smooth)
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        Dice_IOU = IOU_loss + dice_loss

        return Dice_IOU


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    """
    Forward pass of the IoULoss module 
    | Params:
    |  - inputs: Predicted segmentation tensor
    |  - targets: Ground truth segmentation tensor
    |  - smooth: Smooth factor
    | Returns:
    |  - IoU: Loss value
    """

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.flatten()
        targets = targets.flatten()

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


class Focal(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Focal, self).__init__()

    """
    Forward function for Focal module
    | Params:
    |  - inputs: Predicted segmentation tensor
    |  - targets: Ground truth segmentation tensor
    |  - alpha: Weighting factor for balancing positive and negative samples
    |  - gamma: Modulating factor to emphasize hard samples
    |  - smooth: Smoothing factor
    | Returns:
    |  - focal_loss: Loss valuee
    """

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.flatten()
        targets = targets.flatten()

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss
    