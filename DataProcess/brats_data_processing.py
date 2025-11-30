# dataset_brats_png.py
import os
from glob import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BraTSSlicesDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transforms=None, multi_class=True):
        self.images = sorted(glob(os.path.join(images_dir, "*.png")))
        self.masks_dir = masks_dir
        self.transforms = transforms
        self.multi_class = multi_class

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        fname = os.path.basename(img_path)
        mask_path = os.path.join(self.masks_dir, fname)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)   # (H,W)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) # preserve label values

        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        else:
            img = img.astype(np.float32)/255.0
            img = np.expand_dims(img, 0)  # CHW
            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask).long()

        return img, mask

def get_transforms(img_size=256, is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.4),
            A.RandomBrightnessContrast(p=0.4),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])
