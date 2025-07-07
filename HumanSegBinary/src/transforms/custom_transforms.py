# src/transforms/custom_transforms.py

import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import torch


class CustomTransforms:
    def __init__(self, is_train=True, image_size=(480, 640)):
        self.is_train = is_train
        self.image_size = image_size

    def __call__(self, image, mask):
        # Resize to fixed dimensions
        image = TF.resize(image, self.image_size)
        mask = TF.resize(mask, self.image_size)

        if self.is_train:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # Random rotation
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        # Convert to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # Ensure mask is binary (0 or 1)
        mask = (mask > 0.5).float()

        return {"image": image, "mask": mask}
