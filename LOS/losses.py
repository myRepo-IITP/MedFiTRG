import torch
import torch.nn as nn
import torch.nn.functional as F

class RobustFocalLoss(nn.Module):
    """Focal Loss for multi-class classification (LOS)"""
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None, reduction='mean'):
        super(RobustFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes).float()
        probs = F.softmax(inputs, dim=1)
        probs = torch.clamp(probs, 1e-7, 1.0 - 1e-7)
        
        ce_loss = -targets_one_hot * torch.log(probs)
        focal_weight = (1 - probs) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss
        
        if self.class_weights is not None:
            # Ensure weights match the input device
            weights = self.class_weights.unsqueeze(0).expand(targets.size(0), -1)
            class_weights = torch.sum(targets_one_hot * weights, dim=1, keepdim=True)
            focal_loss = focal_loss * class_weights
        
        focal_loss = torch.sum(focal_loss, dim=1)
        
        if self.reduction == 'mean': 
            return focal_loss.mean()
        elif self.reduction == 'sum': 
            return focal_loss.sum()
        return focal_loss