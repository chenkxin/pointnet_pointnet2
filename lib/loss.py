import torch
import torch.nn as nn

from lib.models.base_caps2cnn import CapsuleLoss, Capsule_recon
from lib.models.block import BaseModel



class MarginLoss(BaseModel):
    """Combine margin loss of capsule network."""

    def __init__(self, upper_bound=0.9, lower_bound=0.1, lmda=0.5, **kwargs):
        super(MarginLoss, self).__init__()
        self.upper = upper_bound
        self.lower = lower_bound
        self.lmda = lmda
        self.nclasses = 10 if "nclasses" not in kwargs else kwargs["nclasses"]

    def forward(self, input, target):
        """
        Args:
            input: :math:`(N, C)` where `C = number of classes`,
            target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
              :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
              K-dimensional loss.

        Returns:

        """
        target = (
            torch.eye(self.nclasses)
            .index_select(dim=0, index=target.cpu())
            .to(input.device)
        )  # (N)
        left = (self.upper - input).relu() ** 2  # True negative
        right = (input - self.lower).relu() ** 2  # False positive
        margin_loss = torch.sum(target * left) + self.lmda * torch.sum(
            (1 - target) * right
        )
        return margin_loss


def makeLoss(name, **kwargs):
    if name == "nll":
        return nn.NLLLoss()
    elif name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif name =="CapsuleLoss":
        return CapsuleLoss()
    elif name =="Capsule_recon":
        return Capsule_recon()
