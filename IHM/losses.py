import torch
import torch.nn as nn
import torch.nn.functional as F

class RobustFocalLoss(nn.Module):
    """
    Modified for binary classification - In-Hospital Mortality (IHM)
    """
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None, reduction='mean'):
        super(RobustFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, inputs, targets):
        # For binary classification, ensure proper shape
        if inputs.dim() > 1 and inputs.shape[1] == 1:
            inputs = inputs.squeeze(1)
        if targets.dim() > 1 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        
        # Clip probabilities for numerical stability
        p = torch.sigmoid(inputs)
        p = torch.clamp(p, 1e-7, 1.0 - 1e-7)
        
        # Binary Cross Entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Focal Term: (1 - p_t)^gamma
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_term = (1 - p_t) ** self.gamma
        
        # Alpha balancing
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = alpha_t * focal_term * bce_loss
        
        # Apply positive class weight if provided
        if self.pos_weight is not None:
            class_weights = 1.0 + (targets * (self.pos_weight - 1.0))
            loss = loss * class_weights

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss