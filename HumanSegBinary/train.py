import yaml
import os
import torch
import torch.optim as optim
from tqdm import tqdm
from src.data.human_dataset import DataLoaderFactory
from src.transforms.custom_transforms import CustomTransforms
from src.models import UNet
from src.utils import dice_score, iou_score, accuracy

# Import the professional loss function (save the above code as loss_functions.py)
from loss_functions import CombinedSegmentationLoss, get_professional_loss_and_optimizer

# Load configuration
try:
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError("config.yaml not found. Please create it with required fields.")

# Validate config
required_keys = ["data_root", "batch_size", "num_epochs", "learning_rate", "input_size"]
for key in required_keys:
    if key not in cfg:
        raise KeyError(f"Missing '{key}' in config.yaml")

device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

def train():
    # Build dataloaders
    train_loader = DataLoaderFactory.get_train_dataloader(
        data_root=cfg["data_root"],
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 4),
        transform=CustomTransforms(is_train=True, image_size=(cfg["input_size"][0], cfg["input_size"][1]))
    )
    val_loader = DataLoaderFactory.get_val_dataloader(
        data_root=cfg["data_root"],
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 4),
        transform=CustomTransforms(is_train=False, image_size=(cfg["input_size"][0], cfg["input_size"][1]))
    )

    # Model setup
    model = UNet(n_channels=cfg.get("in_channels", 3), n_classes=cfg.get("num_classes", 1)).to(device)
    
    # Professional loss function and optimizer with momentum
    loss_fn, optimizer, scheduler = get_professional_loss_and_optimizer(model, cfg)
    
    # Alternative: if you prefer the original optimizer setup, use:
    # loss_fn = CombinedSegmentationLoss()
    # optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    best_dice = 0.0
    os.makedirs("checkpoints", exist_ok=True)
    
    # Loss tracking for detailed monitoring - use flexible keys
    train_losses = {}
    val_losses = {}

    # Outer progress bar: epochs
    epoch_bar = tqdm(range(cfg["num_epochs"]), desc="Epochs", unit="epoch")
    for epoch in epoch_bar:
        # Training
        model.train()
        running_loss, running_dice, running_iou = 0.0, 0.0, 0.0
        running_loss_components = {}
        
        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['num_epochs']}", leave=False, unit="batch")
        for images, masks in batch_bar:
            images = images.to(device)
            masks = masks.to(device).float()

            optimizer.zero_grad()
            outputs = model(images)
            
            # Use professional loss function
            loss, loss_dict = loss_fn(outputs, masks)
            loss.backward()
            
            # Gradient clipping for stability (professional practice)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            # Metrics
            with torch.no_grad():
                d = dice_score(outputs, masks)
                i = iou_score(outputs, masks)
            
            running_loss += loss.item()
            running_dice += d
            running_iou += i
            
            # Track loss components - dynamically handle any keys returned
            for key, value in loss_dict.items():
                if key not in running_loss_components:
                    running_loss_components[key] = 0.0
                running_loss_components[key] += value

            # Dynamic postfix based on available loss components
            postfix = {
                "loss": f"{loss.item():.4f}", 
                "dice": f"{d:.4f}", 
                "iou": f"{i:.4f}"
            }
            # Add available loss components to postfix
            for key, value in loss_dict.items():
                if len(postfix) < 6:  # Limit display to avoid clutter
                    postfix[key[:8]] = f"{value:.4f}"  # Truncate key names if too long
            
            batch_bar.set_postfix(postfix)

        # Update learning rate scheduler
        scheduler.step()

        # Epoch summary (training)
        avg_loss = running_loss / len(train_loader)
        avg_dice = running_dice / len(train_loader)
        avg_iou = running_iou / len(train_loader)
        
        # Average loss components and store for tracking
        for key in running_loss_components:
            avg_component = running_loss_components[key] / len(train_loader)
            if key not in train_losses:
                train_losses[key] = []
            train_losses[key].append(avg_component)

        # Validation
        model.eval()
        val_loss, val_dice, val_iou = 0.0, 0.0, 0.0
        val_loss_components = {}
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device).float()
                outputs = model(images)
                
                loss, loss_dict = loss_fn(outputs, masks)
                val_loss += loss.item()
                val_dice += dice_score(outputs, masks)
                val_iou += iou_score(outputs, masks)
                
                # Track validation loss components - dynamically handle keys
                for key, value in loss_dict.items():
                    if key not in val_loss_components:
                        val_loss_components[key] = 0.0
                    val_loss_components[key] += value

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        
        # Average validation loss components and store for tracking
        for key in val_loss_components:
            avg_component = val_loss_components[key] / len(val_loader)
            if key not in val_losses:
                val_losses[key] = []
            val_losses[key].append(avg_component)

        # Save best model
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            save_path = os.path.join("checkpoints", "unet_best.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_dice': best_dice,
                'config': cfg
            }, save_path)
            print(f"\nBest model saved to {save_path} with val_dice: {best_dice:.4f}")

        # Detailed logging every 10 epochs
        if (epoch + 1) % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"\nEpoch {epoch+1} Detailed Loss Breakdown:")
            print(f"  Learning Rate: {current_lr:.2e}")
            
            # Print train losses
            train_loss_str = "  Train - "
            for key in train_losses:
                train_loss_str += f"{key}: {train_losses[key][-1]:.4f}, "
            print(train_loss_str.rstrip(", "))
            
            # Print validation losses
            val_loss_str = "  Val   - "
            for key in val_losses:
                val_loss_str += f"{key}: {val_losses[key][-1]:.4f}, "
            print(val_loss_str.rstrip(", "))

        # Update epoch bar
        epoch_bar.set_postfix({
            "train_loss": f"{avg_loss:.4f}",
            "train_dice": f"{avg_dice:.4f}",
            "train_iou": f"{avg_iou:.4f}",
            "val_loss": f"{avg_val_loss:.4f}",
            "val_dice": f"{avg_val_dice:.4f}",
            "val_iou": f"{avg_val_iou:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })

    # Save final model with complete state
    save_path = os.path.join("checkpoints", "unet_epoch_final.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': cfg["num_epochs"],
        'final_dice': avg_val_dice,
        'config': cfg,
        'train_losses': train_losses,
        'val_losses': val_losses
    }, save_path)
    print(f"\nTraining complete. Final model saved to {save_path}")
    print(f"Best validation Dice score: {best_dice:.4f}")

if __name__ == "__main__":
    train()