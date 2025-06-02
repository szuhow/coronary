import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def compute_loss(masks_pred, true_masks, model):
    """ Computes the combined loss function dynamically """
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    
    if model.n_classes == 1:
        # Binary case (logits -> BCE + Dice)
        bce = criterion(masks_pred.squeeze(1), true_masks.float())
        dice = dice_loss(torch.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
        loss = 0.8 * bce + 0.2 * dice  # Adjust weight dynamically if needed
    else:
        # Multi-class case (logits -> CE + Dice)
        ce_loss = criterion(masks_pred, true_masks)
        dice = dice_loss(
            masks_pred.softmax(dim=1),  # Use softmax directly
            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
            multiclass=True
        )
        loss = 0.8 * ce_loss + 0.2 * dice  # Adjust weight dynamically
    
    return loss