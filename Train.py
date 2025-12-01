import copy
import random


import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchvision.datasets import *
from torchvision.transforms import *

from SegmentationPrune.Models.Unet import UNet
from .Utills.TrainingModulesUtills import evaluate
from .DataProcess.DataPreprocessing import get_dataloaders
from .Utills.TrainingModulesUtills import Training
from .Utills.ViewerUtills import plot_accuracy, plot_loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:",device)

seed=0
random.seed(seed)
basedir='SegmentationPrune'
select_model='Unet'
img_path, mask_path = "dataset/brats2d/images", "dataset/brats2d/masks"
train_dataloader,test_dataloader=get_dataloaders(img_path, mask_path, batch_size=64)


model=UNet(in_channels=1, out_channels=1)

# ########load from path only for retraining #####
# model_path='checkpoint/Resnet-34/Resnet-34_cifar_95.66999816894531.pth'
# model = torch.load(model_path, map_location=torch.device(device))  # Use 'cpu' if necessary
################################################
model = model.to(device)


criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=1e-4)

num_epochs=200
scheduler = CosineAnnealingLR(optimizer, num_epochs)
# scheduler = CosineAnnealingLR(optimizer, T_max=50)

best_model, train_losses, test_losses, dices = Training(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=20,        # choose epochs
    scheduler=None
)

model=copy.deepcopy(best_model)
metric,_ = evaluate(model, test_dataloader)
print(f"Best model accuray:", metric)

plot_accuracy(dices)
plot_loss(train_losses,test_losses)

torch.save(model, f'{basedir}/checkpoint/{select_model}_cifar_{metric}.pth')
torch.save(model.state_dict(), f'{basedir}/checkpoint/{select_model}_cifar_{metric:0.2f}_state_dict.pth')

    
