import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class HumanSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))])
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg'))])

        assert len(self.images) == len(self.masks), "Mismatch between images and masks count."

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # binary mask

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        return image, mask

class DataLoaderFactory:
    @staticmethod
    def get_train_dataloader(data_root, batch_size, num_workers=4, transform=None, shuffle=True):
        image_dir = os.path.join(data_root, "train", "images")
        mask_dir = os.path.join(data_root, "train", "masks")
        dataset = HumanSegmentationDataset(image_dir, mask_dir, transform=transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    @staticmethod
    def get_val_dataloader(data_root, batch_size, num_workers=4, transform=None, shuffle=False):
        image_dir = os.path.join(data_root, "test", "images")  # Using 'test' as validation for now
        mask_dir = os.path.join(data_root, "test", "masks")
        dataset = HumanSegmentationDataset(image_dir, mask_dir, transform=transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    @staticmethod
    def get_test_dataloader(data_root, batch_size, num_workers=4, transform=None, shuffle=False):
        image_dir = os.path.join(data_root, "test", "images")
        mask_dir = os.path.join(data_root, "test", "masks")
        dataset = HumanSegmentationDataset(image_dir, mask_dir, transform=transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)