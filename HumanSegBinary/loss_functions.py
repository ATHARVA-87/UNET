import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    """Dice Loss for better overlap optimization"""
    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid to logits
        inputs = torch.sigmoid(inputs)
        
        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice_score = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice_score

class TverskyLoss(nn.Module):
    """Tversky Loss - generalization of Dice Loss with controllable precision/recall balance"""
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # Weight for false positives
        self.beta = beta    # Weight for false negatives
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.sigmoid(inputs)
        
        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        true_pos = (inputs * targets).sum()
        false_neg = (targets * (1 - inputs)).sum()
        false_pos = ((1 - targets) * inputs).sum()
        
        tversky_score = (true_pos + self.smooth) / (true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth)
        
        return 1 - tversky_score

class BoundaryLoss(nn.Module):
    """Boundary Loss for better edge detection"""
    def __init__(self, theta0: float = 3.0, theta: float = 5.0):
        super(BoundaryLoss, self).__init__()
        self.theta0 = theta0
        self.theta = theta

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Simplified boundary loss focusing on edge regions
        """
        inputs = torch.sigmoid(inputs)
        
        # Create edge maps using gradient
        def get_edges(tensor):
            # Sobel operators
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tensor.dtype, device=tensor.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tensor.dtype, device=tensor.device).view(1, 1, 3, 3)
            
            grad_x = F.conv2d(tensor, sobel_x, padding=1)
            grad_y = F.conv2d(tensor, sobel_y, padding=1)
            
            return torch.sqrt(grad_x**2 + grad_y**2)
        
        target_edges = get_edges(targets)
        input_edges = get_edges(inputs)
        
        # Focus loss on edge regions
        edge_weight = (target_edges > 0.1).float() * 2.0 + 1.0
        edge_loss = F.mse_loss(input_edges * edge_weight, target_edges * edge_weight)
        
        return edge_loss

class CombinedSegmentationLoss(nn.Module):
    """
    Professional-grade combined loss function for human binary segmentation.
    Optimized for limited training data and precise boundary detection.
    """
    def __init__(self, 
                 focal_weight: float = 0.4,
                 dice_weight: float = 0.3,
                 tversky_weight: float = 0.2,
                 boundary_weight: float = 0.1,
                 focal_alpha: float = 1.0,
                 focal_gamma: float = 2.0,
                 tversky_alpha: float = 0.3,
                 tversky_beta: float = 0.7,
                 smooth: float = 1e-6):
        """
        Args:
            focal_weight: Weight for focal loss (handles class imbalance)
            dice_weight: Weight for dice loss (optimizes overlap)
            tversky_weight: Weight for tversky loss (precision/recall balance)
            boundary_weight: Weight for boundary loss (edge precision)
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter (higher = more focus on hard examples)
            tversky_alpha: False positive weight in Tversky loss
            tversky_beta: False negative weight in Tversky loss
        """
        super(CombinedSegmentationLoss, self).__init__()
        
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.boundary_weight = boundary_weight
        
        # Initialize loss components
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss(smooth=smooth)
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta, smooth=smooth)
        self.boundary_loss = BoundaryLoss()
        
        # Validate weights sum to 1
        total_weight = focal_weight + dice_weight + tversky_weight + boundary_weight
        if abs(total_weight - 1.0) > 1e-6:
            print(f"Warning: Loss weights sum to {total_weight}, not 1.0. Consider normalizing.")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            inputs: Raw logits from model [B, 1, H, W]
            targets: Ground truth masks [B, 1, H, W] with values in [0, 1]
        
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary with individual loss components for monitoring
        """
        # Compute individual losses
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        tversky = self.tversky_loss(inputs, targets)
        boundary = self.boundary_loss(inputs, targets)
        
        # Combine losses
        total_loss = (self.focal_weight * focal + 
                     self.dice_weight * dice + 
                     self.tversky_weight * tversky + 
                     self.boundary_weight * boundary)
        
        # Return detailed breakdown for monitoring
        loss_dict = {
            'total_loss': total_loss.item(),
            'focal_loss': focal.item(),
            'dice_loss': dice.item(),
            'tversky_loss': tversky.item(),
            'boundary_loss': boundary.item()
        }
        
        return total_loss, loss_dict

# Example usage and integration
def get_professional_loss_and_optimizer(model, config):
    """
    Get the professional loss function and optimizer with momentum.
    Recommended for human segmentation tasks.
    """
    # Loss function optimized for human segmentation
    loss_fn = CombinedSegmentationLoss(
        focal_weight=0.4,      # High weight for class imbalance handling
        dice_weight=0.3,       # Good overlap optimization
        tversky_weight=0.2,    # Precision/recall balance
        boundary_weight=0.1,   # Edge refinement
        focal_gamma=2.0,       # Focus on hard examples
        tversky_alpha=0.3,     # Less penalty for false positives
        tversky_beta=0.7       # More penalty for false negatives (missing humans)
    )
    
    # Optimizer with momentum and weight decay for better generalization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(0.9, 0.999),           # Momentum parameters
        weight_decay=1e-4,            # L2 regularization
        eps=1e-8
    )
    
    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,                       # Initial restart period
        T_mult=2,                     # Period multiplication factor
        eta_min=1e-6                  # Minimum learning rate
    )
    
    return loss_fn, optimizer, scheduler

# Alternative: Polynomial learning rate decay (also excellent choice)
def get_poly_lr_scheduler(optimizer, max_epochs, power=0.9):
    """Polynomial learning rate decay - very effective for segmentation"""
    lambda_func = lambda epoch: (1 - epoch / max_epochs) ** power
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)