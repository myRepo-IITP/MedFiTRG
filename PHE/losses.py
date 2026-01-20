import torch
import torch.nn as nn
import torch.nn.functional as F

class RobustFocalLoss(nn.Module):
    """Focal Loss for multi-label classification (Phenotypes)"""
    def __init__(self, alpha=0.25, gamma=2.0, pos_weights=None, reduction='mean'):
        super(RobustFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer('pos_weights', pos_weights)

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        p = torch.clamp(p, 1e-7, 1.0 - 1e-7)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_term = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal_term * bce_loss
        
        if self.pos_weights is not None:
            class_weights = 1.0 + (targets * (self.pos_weights - 1.0))
            loss = loss * class_weights
            
        if self.reduction == 'mean': 
            return loss.mean()
        elif self.reduction == 'sum': 
            return loss.sum()
        return loss