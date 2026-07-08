import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance.

    Args:
        alpha: Per-class weights tensor of shape (num_classes,)
        gamma: Focusing parameter. Higher values down-weight easy examples.
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class MLP(nn.Module):
    """Multi-layer perceptron for per-residue binary classification.

    Args:
        in_features: Input dimension (e.g. 320 for ESM2-8M)
        hidden_size: Hidden layer dimension
        num_layers: Number of hidden layers
        num_classes: Number of output classes (2 for binary)
    """

    def __init__(self, in_features: int, hidden_size: int, num_layers: int, num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.final = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.softmax(self.final(x))
