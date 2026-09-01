"""PyTorch training adapter for baseline dense neural networks."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_DEVICE = "cuda"
REG_STRENGTH = 1e-4


def resolve_device(device=DEFAULT_DEVICE):
    """Return the requested device; fail fast when CUDA is required but unavailable."""
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required but not available. Install a CUDA-enabled PyTorch build "
            "and ensure the GPU is visible to this environment."
        )
    return torch.device(device)


def create_optimizer(model, optim_name, lr=1e-3):
    """Map legacy optimizer names to torch.optim classes."""
    optimizers = {
        "Adadelta": torch.optim.Adadelta,
        "Adagrad": torch.optim.Adagrad,
        "Adam": torch.optim.Adam,
        "Adamax": torch.optim.Adamax,
        "Nadam": torch.optim.NAdam,
        "RMSprop": torch.optim.RMSprop,
        "SGD": torch.optim.SGD,
    }
    if optim_name not in optimizers:
        raise ValueError(f"Unsupported optimizer: {optim_name}")
    return optimizers[optim_name](model.parameters(), lr=lr)


def _tensor_dataset(X, y):
    x_tensor = torch.as_tensor(X, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.float32).view(-1)
    return TensorDataset(x_tensor, y_tensor)


def _regularization_penalty(model, kernel_reg=None, bias_reg=None):
    """Apply explicit L1/L2 penalties mapped from the former Keras regularizer strings."""
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        reg_kind = kernel_reg if "weight" in name else bias_reg if "bias" in name else None
        if reg_kind is None:
            continue
        if reg_kind in ("L1", "L1L2"):
            penalty = penalty + REG_STRENGTH * param.abs().sum()
        if reg_kind in ("L2", "L1L2"):
            penalty = penalty + REG_STRENGTH * param.pow(2).sum()
    return penalty


def _batch_accuracy(logits, targets):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    return (preds == targets).float().mean().item()


def train_model(
    model,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    epochs=200,
    batch_size=32,
    kernel_reg=None,
    bias_reg=None,
    optim="Adam",
    device=DEFAULT_DEVICE,
    return_history=False,
):
    """Train a model with BCEWithLogitsLoss; sigmoid is deferred to prediction/metrics."""
    print(f"Training model on {device}")
    device = resolve_device(device)
    model = model.to(device)
    optimizer = create_optimizer(model, optim)
    # Numerically stable binary cross-entropy: expects raw logits, not sigmoid outputs.
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(_tensor_dataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = None
    if X_val is not None and y_val is not None:
        val_loader = DataLoader(_tensor_dataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    history = {"loss": [], "binary_accuracy": [], "val_loss": [], "val_binary_accuracy": []}

    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        batches = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y) + _regularization_penalty(model, kernel_reg, bias_reg)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += _batch_accuracy(logits.detach(), batch_y)
            batches += 1

        if return_history:
            history["loss"].append(epoch_loss / max(batches, 1))
            history["binary_accuracy"].append(epoch_acc / max(batches, 1))

            if val_loader is not None:
                val_loss, val_acc = _evaluate_loader(model, val_loader, criterion, device, kernel_reg, bias_reg)
                history["val_loss"].append(val_loss)
                history["val_binary_accuracy"].append(val_acc)

    if return_history:
        return model, history
    return model


def _evaluate_loader(model, data_loader, criterion, device, kernel_reg=None, bias_reg=None):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    batches = 0
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y) + _regularization_penalty(model, kernel_reg, bias_reg)
            total_loss += loss.item()
            total_acc += _batch_accuracy(logits, batch_y)
            batches += 1
    model.train()
    return total_loss / max(batches, 1), total_acc / max(batches, 1)


@torch.no_grad()
def predict_proba(model, X, device=DEFAULT_DEVICE, batch_size=32):
    """Return class-1 probabilities by applying sigmoid to logits at inference time."""
    device = resolve_device(device)
    model = model.to(device)
    model.eval()
    x_tensor = torch.as_tensor(X, dtype=torch.float32)
    probs = []
    for start in range(0, len(x_tensor), batch_size):
        batch = x_tensor[start : start + batch_size].to(device)
        logits = model(batch)
        probs.append(torch.sigmoid(logits).cpu())
    return torch.cat(probs).numpy()


@torch.no_grad()
def predict_classes(model, X, device=DEFAULT_DEVICE, batch_size=32, threshold=0.5):
    """Threshold sigmoid probabilities for balanced-accuracy scoring."""
    probs = predict_proba(model, X, device=device, batch_size=batch_size)
    return (probs > threshold).astype("int32")
