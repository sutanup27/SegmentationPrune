# dataset_brats_png.py
import os
from glob import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# brats_to_png.py
import os
from glob import glob
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm

ROOT = "dataset\Brasts_TrainingData"               # root with patient folders (e.g. Brats/BraTS-GLI-00001/...)
OUT_ROOT = "brats2d"        # creates brats2d/images, brats2d/masks
MODALITY = "flair"          # choose modality to export
IMG_SIZE = 256
MIN_TUMOR_PIXELS = 5        # skip slices with tiny/no tumor

os.makedirs(os.path.join(OUT_ROOT, "images"), exist_ok=True)
os.makedirs(os.path.join(OUT_ROOT, "masks"), exist_ok=True)

def normalize_to_uint8(img):
    p1, p99 = np.percentile(img, (1,99))
    img = np.clip(img, p1, p99)
    img = (img - p1) / (p99 - p1 + 1e-8)
    img = (img * 255).astype(np.uint8)
    return img

def remap_mask(seg):
    # remap {0,1,2,4} -> {0,1,2,3}
    out = np.zeros_like(seg, dtype=np.uint8)
    out[seg == 0] = 0
    out[seg == 1] = 1
    out[seg == 2] = 2
    out[seg == 4] = 3
    return out

print("Processing BraTS dataset...")
patients = sorted([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))])
count = 0
MODALITY_MAP = {
    "flair": "t2f",
    "t1": "t1n",
    "t1ce": "t1c",
    "t2": "t2w",
}

for p in tqdm(patients, desc="patients"):
    pdir = os.path.join(ROOT, p)
    # Map modality name to your dataset suffix

    mod_suffix = MODALITY_MAP[MODALITY]   # e.g., flair → t2f

    mod_path = os.path.join(pdir, f"{p}-{mod_suffix}.nii.gz")
    seg_path = os.path.join(pdir, f"{p}-seg.nii.gz")
    vol = nib.load(mod_path).get_fdata()
    seg = nib.load(seg_path).get_fdata().astype(np.uint8)
    seg = remap_mask(seg)

    D = vol.shape[2]
    for i in range(D):
        mask_slice = seg[:, :, i]
        if (mask_slice > 0).sum() < MIN_TUMOR_PIXELS:
            continue
        img_slice = vol[:, :, i]
        img_u8 = normalize_to_uint8(img_slice)
        img_rs = cv2.resize(img_u8, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        mask_rs = cv2.resize(mask_slice, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        name = f"{p}_slice_{i:03d}.png"
        cv2.imwrite(os.path.join(OUT_ROOT, "images", name), img_rs)
        cv2.imwrite(os.path.join(OUT_ROOT, "masks", name), mask_rs)
        count += 1

print(f"Saved {count} slices to {OUT_ROOT}")
