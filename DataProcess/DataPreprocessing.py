import random
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
import os
from glob import glob
import cv2
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BraTSSlicesDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transforms=None, multi_class=True):
        self.images = sorted(glob.glob(os.path.join(images_dir, "*.png")))
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
        print(np.shape(img))
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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Slows training but ensures reproducibility


# make small helper function to create folder lists (or create txt files)
def make_dataset_from_list(img_list, images_dir, masks_dir, transforms):
    # easiest: make a temporary dataset object which only holds chosen files.
    ds = BraTSSlicesDataset(images_dir, masks_dir, transforms=transforms)
    # override images list
    ds.images = img_list
    return ds
 

def get_datasets(image_path, mask_path,test_size=0.2, img_size=256):
    dataset={}
    all_images = sorted(glob.glob(f"{image_path}/*.png"))
    print(f"Total images found: {len(all_images)}")
    train_imgs, val_imgs = train_test_split(all_images, test_size=test_size, random_state=42)

    dataset["train"] =make_dataset_from_list(train_imgs, image_path, mask_path, get_transforms(img_size, True))
    dataset["test"] = make_dataset_from_list(val_imgs, image_path, mask_path, get_transforms(img_size, False))
    return dataset["train"],dataset["test"]


def get_dataloaders(image_path, mask_path, test_size=0.2, img_size=256, batch_size=512):
    train_ds, val_ds = get_datasets( image_path, mask_path, img_size=img_size, test_size=test_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader,val_loader