
from PruningNAS.Models.Unet import *
from torch import nn

from PruningNAS.Utills.Utill import get_sparsity_dic_template, print_model
from PruningNAS.Utills.ViewerUtills import accumulate_plot_figures

model=UNet(in_channels=1, out_channels=1)

for name, module in model.named_modules():
    # Skip top-level module
    if name == "":
        continue

    # If the module has weights
    if hasattr(module, "weight") and module.weight is not None:
        print(f"{name} -> weight: {tuple(module.weight.shape)}")

    # If the module has bias
    if hasattr(module, "bias") and module.bias is not None:
        print(f"{name} -> bias:   {tuple(module.bias.shape)}")
