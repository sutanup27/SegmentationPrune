from torch.optim import *
from torch.optim.lr_scheduler import *
from torchvision.datasets import *
from torchvision.transforms import *
from DataPreprocessing import  get_dataloaders, get_datasets
import torch
import numpy as np

img_path, mask_path = "dataset/brats2d/images", "dataset/brats2d/masks"



  # Visualize the image
def inspect_batch(batch_images, batch_masks, name="batch"):
    # batch_images: (B,C,H,W), batch_masks: (B,...) 
    B = batch_images.shape[0]
    print(f"{name}: images {tuple(batch_images.shape)}, masks {tuple(batch_masks.shape)}")
    for i in range(min(4, B)):
        img = batch_images[i]
        m = batch_masks[i]  
        # shape
        print(f" sample {i}: img_shape={tuple(img.shape)}, mask_shape={tuple(m.shape)}")
        # mask stats
        m_np = m.detach().cpu().numpy()
        unique = np.unique(m_np)
        print(f"   unique mask labels: {unique}")

        if m_np.dtype != np.uint8:
            print(f"   mask dtype:{m_np.dtype}, min/max: {m_np.min()}/{m_np.max()}")


train_dataloader=get_dataloaders(img_path, mask_path, batch_size=8)[0]
img,mask=next(iter(train_dataloader))
inspect_batch(img,mask, name="Train Batch")


