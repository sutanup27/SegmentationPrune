import copy
import random


import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchvision.datasets import *
from torchvision.transforms import *
from .Models.ResNetBasic import *
from .Models.ResNetBottleNeck import *
from .Utills.TrainingModulesUtills import evaluate
from .DataProcess.DataPreprocessing import get_dataloaders
from .Models.VGG import *
from .Utills.TrainingModulesUtills import Training
from .Utills.ViewerUtills import plot_accuracy, plot_loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:",device)

seed=0
random.seed(seed)
basedir='PruningNAS'

path='./dataset/cifar10'
classes=10
train_dataloader,test_dataloader=get_dataloaders(path,batch_size=64)

select_model='Resnet-50'
if select_model=='Vgg-16':
    model=VGG(classes=classes)
elif select_model=='Resnet-18':
    model = ResNet18(classes=classes)
elif select_model=='Resnet-34':
    model = ResNet34(classes=classes)
elif select_model=='Resnet-50':
    model = ResNet50(classes=classes)
elif select_model=='Resnet-101':
    model = ResNet101(classes=classes)
elif select_model=='Resnet-152':
    model = ResNet152(classes=classes)
else:
    print("Model does not exists")
    exit

# ########load from path only for retraining #####
# model_path='checkpoint/Resnet-34/Resnet-34_cifar_95.66999816894531.pth'
# model = torch.load(model_path, map_location=torch.device(device))  # Use 'cpu' if necessary
################################################
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = SGD( model.parameters(), lr=0.001,  momentum=0.9,  weight_decay=5e-4,)

num_epochs=200
scheduler = CosineAnnealingLR(optimizer, num_epochs)
# scheduler = CosineAnnealingLR(optimizer, T_max=50)

best_model, losses, test_losses, accs=Training( model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs=num_epochs,scheduler=scheduler)

model=copy.deepcopy(best_model)
metric,_ = evaluate(model, test_dataloader)
print(f"Best model accuray:", metric)

plot_accuracy(accs)
plot_loss(losses,test_losses)

torch.save(model, f'{basedir}/checkpoint/{select_model}/{select_model}_cifar_{metric}.pth')
torch.save(model.state_dict(), f'{basedir}/checkpoint/{select_model}/{select_model}_cifar_{metric:0.2f}_state_dict.pth')

    
