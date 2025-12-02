import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----- metrics & helpers -----
def dice_coeff_batch(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    probs: (B,1,H,W) floating probabilities in [0,1]
    targets: (B,1,H,W) binary {0,1} floats
    returns mean dice over batch (scalar tensor)
    """
    probs_flat = probs.view(probs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    inter = (probs_flat * targets_flat).sum(dim=1)
    union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
    dice = (2.0 * inter + eps) / (union + eps)
    return dice.mean()


def binarize_logits(logits: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """Logits -> binary mask (B,1,H,W) as float 0/1"""
    probs = torch.sigmoid(logits)
    return (probs >= thr).float(), probs


# ----- train step (binary segmentation) -----
def train(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    epoch: int = 0,
    callbacks=None,
    device=device
) -> tuple:
    """
    Returns (avg_loss, avg_dice)
    """
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    n = 0

    for inputs, targets in tqdm(dataloader, desc=f'train epoch:{epoch}', leave=False):
        # inputs: (B,C,H,W), targets: either (B,1,H,W) float for BCE or (B,H,W) long for CE
        inputs = inputs.to(device, dtype=torch.float32)
        targets = targets.to(device)

        # prepare targets for BCE: ensure float (B,1,H,W)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)  # (B,1,H,W)
        # If criterion is BCEWithLogitsLoss, targets must be float
        if isinstance(criterion, nn.BCEWithLogitsLoss) or hasattr(criterion, 'bce'):
            targets = targets.float()

        optimizer.zero_grad()
        logits = model(inputs)                       # (B,1,H,W)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        # callbacks (pruning mask application etc.) — call after optimizer.step so mask persists
        if callbacks is not None:
            for cb in callbacks:
                cb()

        # metrics
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            dice = dice_coeff_batch(probs, targets.float())
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_dice += dice.item() * batch_size
        n += batch_size

    avg_loss = total_loss / n
    avg_dice = total_dice / n
    return avg_loss, avg_dice


# ----- prediction helper (single image) -----
def predict(model: nn.Module, input_tensor: torch.Tensor, thr: float = 0.5, device=device):
    """
    input_tensor: (C,H,W) or (1,H,W) torch tensor on CPU.
    Returns binary mask (H,W) numpy uint8 and probability map (H,W) float32
    """
    model.to(device)
    model.eval()
    if input_tensor.dim() == 3:
        inp = input_tensor.unsqueeze(0).to(device).float()  # (1,C,H,W)
    else:
        raise ValueError("input_tensor must be (C,H,W)")
    with torch.no_grad():
        logits = model(inp)               # (1,1,H,W)
        probs = torch.sigmoid(logits)
        pred_bin = (probs >= thr).squeeze(0).squeeze(0).cpu().numpy().astype('uint8')
        prob_map = probs.squeeze(0).squeeze(0).cpu().numpy().astype('float32')
    return pred_bin, prob_map


# ----- evaluation (binary segmentation) -----
@torch.inference_mode()
def evaluate(model: nn.Module, dataloader: DataLoader, criterion=None, device=device, verbose=True):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    n = 0

    for inputs, targets in tqdm(dataloader, desc="eval", leave=False, disable=not verbose):
        inputs = inputs.to(device, dtype=torch.float32)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.to(device)
        if criterion is not None and isinstance(criterion, nn.BCEWithLogitsLoss):
            targets = targets.float()

        logits = model(inputs)
        if criterion is not None:
            loss = criterion(logits, targets)
            total_loss += loss.item() * inputs.size(0)

        probs = torch.sigmoid(logits)
        dice = dice_coeff_batch(probs, targets.float())
        total_dice += dice.item() * inputs.size(0)
        n += inputs.size(0)

    avg_dice = total_dice / n if n > 0 else 0.0
    avg_loss = total_loss / n if (criterion is not None and n > 0) else 0.0
    return avg_dice, avg_loss


# ----- high-level Training loop (binary) -----
def Training(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs=10, scheduler=None,checkpoint_path=None):
    losses, val_losses, dices = [], [], []
    best_dice = -1.0
    best_model = None

    model.to(device)
    for epoch in range(1, num_epochs + 1):
        train_loss, train_dice = train(model, train_dataloader, criterion, optimizer, epoch=epoch)
        val_dice, val_loss = evaluate(model, test_dataloader, criterion)
        print(f"Epoch {epoch}: Train loss {train_loss:.4f}, Train Dice {train_dice:.4f} | Val loss {val_loss:.4f}, Val Dice {val_dice:.4f}")
        if scheduler is not None:
            scheduler.step()

        if val_dice > best_dice:
            best_dice = val_dice
            best_model = copy.deepcopy(model)

        if epoch%5==0 and checkpoint_path!=None:
            torch.save(best_model,f"{checkpoint_path}/Unet_brayts_{epoch}_{best_dice}.pth")

        losses.append(train_loss)
        val_losses.append(val_loss)
        dices.append(val_dice)

    return best_model, losses, val_losses, dices


# ----- fine-tuning pruned model -----
def TrainingPrunned(pruned_model, train_dataloader, test_dataloader, criterion, optimizer, pruner,
                    scheduler=None, num_finetune_epochs=5, isCallback=True):
    best_accuracy = 0.0
    best_model = None
    pruned_model.to(device)

    for epoch in range(num_finetune_epochs):
        callbacks = [lambda: pruner.apply(pruned_model)] if isCallback else None
        train_loss, train_dice = train(pruned_model, train_dataloader, criterion, optimizer, epoch=epoch + 1, callbacks=callbacks)
        val_dice, val_loss = evaluate(pruned_model, test_dataloader, criterion)
        print(f"Finetune Epoch {epoch+1}: Train Dice {train_dice:.4f}, Val Dice {val_dice:.4f}, Train loss {train_loss:.4f}, Val loss {val_loss:.4f}")

        if val_dice > best_accuracy:
            best_accuracy = val_dice
            best_model = copy.deepcopy(pruned_model)

        if scheduler is not None:
            scheduler.step()

    return best_accuracy, best_model
