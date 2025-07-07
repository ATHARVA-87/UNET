# test.py

import yaml
import os
import torch
from tqdm import tqdm

from src.data.human_dataset import DataLoaderFactory
from src.transforms.custom_transforms import CustomTransforms
from src.models import UNet
from src.utils import accuracy, dice_score, iou_score

# Load configuration
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

def evaluate():
    # build test dataloader
    test_loader = DataLoaderFactory.get_test_dataloader(
        data_root=cfg["data_root"],
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 4),
        transform=CustomTransforms(is_train=False,
                                   image_size=(cfg["input_size"][0], cfg["input_size"][1]))
    )

    # load model
    model = UNet(n_channels=cfg.get("in_channels", 3),
                 n_classes=cfg.get("num_classes", 1)).to(device)

    # assume the final checkpoint is saved here:
    ckpt_path = cfg.get("model_save_path", "checkpoints/unet_epoch_final.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # accumulate metrics
    total_acc = 0.0
    total_dice = 0.0
    total_iou  = 0.0
    n_batches  = 0

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            masks  = masks.to(device).float()

            outputs = model(images)

            total_acc  += accuracy(outputs, masks)
            total_dice += dice_score(outputs, masks)
            total_iou  += iou_score(outputs, masks)
            n_batches  += 1

    avg_acc  = total_acc  / n_batches
    avg_dice = total_dice / n_batches
    avg_iou  = total_iou  / n_batches

    # write results to text file
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "evaluation_metrics.txt")
    with open(out_path, "w") as f:
        f.write(f"Evaluation Results\n")
        f.write(f"===================\n")
        f.write(f"Average Accuracy : {avg_acc:.4f}\n")
        f.write(f"Average Dice     : {avg_dice:.4f}\n")
        f.write(f"Average IoU      : {avg_iou:.4f}\n")

    print(f"\nDone. Metrics saved to {out_path}")

if __name__ == "__main__":
    evaluate()
