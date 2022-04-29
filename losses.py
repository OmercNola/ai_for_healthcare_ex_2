ALPHA = 0.8
GAMMA = 2
import torch.nn.functional as F
import torch

def dice_score(inputs, targets, smooth=1):
    # flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice_score = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

    return dice_score


def get_dice_loss(inputs, targets, smooth=1):
    # flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

    return dice_loss


def get_focal_loss(inputs, targets, alpha=0.8, gamma=2, smooth=1):
    # flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    # first compute binary cross-entropy
    BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
    BCE_EXP = torch.exp(-BCE)
    focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

    return focal_loss


def combo_loss(inputs, targets):
    dice_loss = get_dice_loss(inputs, targets)
    BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
    focal_loss = get_focal_loss(inputs, targets)

    return 1 * dice_loss + 4 * focal_loss + 3 * BCE