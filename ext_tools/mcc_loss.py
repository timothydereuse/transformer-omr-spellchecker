import torch
import torch.nn as nn
from torch.nn import functional as F

# Taken from:
# https://github.com/kakumarabhishek/MCC-Loss/blob/main/loss.py

class Dice_Loss(nn.Module):
    """
    Calculates the Sørensen-Dice coefficient-based loss.
    Taken from
    https://github.com/SaoYan/IPMI2019-AttnMel/blob/master/loss.py#L28

    Args:
        inputs (torch.Tensor): 1-hot encoded predictions
        targets (torch.Tensor): 1-hot encoded ground truth
    """

    def __init__(self):
        super(Dice_Loss, self).__init__()

    def forward(self, inputs, targets):
        """
        Dice(A, B) = (2 * |intersection(A, B)|) / (|A| + |B|)
        where |x| denotes the cardinality of the set x.
        """
        mul = torch.mul(inputs, targets)
        add = torch.add(inputs, 1, targets)
        dice = 2 * torch.div(mul.sum(), add.sum())
        return 1 - dice


class MCC_Loss(nn.Module):
    """
    Calculates the proposed Matthews Correlation Coefficient-based loss.

    Args:
        inputs (torch.Tensor): 1-hot encoded predictions
        targets (torch.Tensor): 1-hot encoded ground truth
    """

    def __init__(self):
        super(MCC_Loss, self).__init__()

    def forward(self, inputs, targets):
        """
        MCC = (TP.TN - FP.FN) / sqrt((TP+FP) . (TP+FN) . (TN+FP) . (TN+FN))
        where TP, TN, FP, and FN are elements in the confusion matrix.
        """

        inputs = torch.sigmoid(inputs)

        tp = torch.sum(torch.mul(inputs, targets))
        tn = torch.sum(torch.mul((1 - inputs), (1 - targets)))
        fp = torch.sum(torch.mul(inputs, (1 - targets)))
        fn = torch.sum(torch.mul((1 - inputs), targets))

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        denominator = torch.sqrt(
            torch.add(tp, fp)
            * torch.add(tp, fn)
            * torch.add(tn, fp)
            * torch.add(tn, fn)
        )

        # Adding 1 to the denominator to avoid divide-by-zero errors.
        mcc = torch.div(numerator.sum(), denominator.sum() + 1.0)
        return 1 - mcc