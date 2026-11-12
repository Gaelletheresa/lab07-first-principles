import copy

import torch
from torch import nn

from model import RentalNet
from prepare import make_loaders


def run_epoch(model, loader, loss_fn, optimizer=None):
    """Run one epoch. Trains if given an optimizer, otherwise only measures.

    Returns the mean loss and the mean absolute error (in rentals).
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_abs_error = 0.0
    total_rows = 0

    with torch.set_grad_enabled(training):
        for features, targets in loader:
            predictions = model(features)
            loss = loss_fn(predictions, targets)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            rows = len(targets)
            total_loss += loss.item() * rows
            total_abs_error += (predictions - targets).abs().sum().item()
            total_rows += rows

    return total_loss / total_rows, total_abs_error / total_rows


def train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs):
    """Train for many epochs, log every epoch, restore the best weights.

    Returns (history, best_epoch). The epochs are numbered from 1.
    """
    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        train_loss, _ = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_mae = run_epoch(model, val_loader, loss_fn)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae,
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return history, best_epoch


if __name__ == "__main__":
    torch.manual_seed(2026)
    train_loader, val_loader, _ = make_loaders()

    model = RentalNet()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    history, best_epoch = train_model(
        model, train_loader, val_loader, loss_fn, optimizer, epochs=30
    )
    for row in history:
        print(
            f"epoch {row['epoch']:3d}  train {row['train_loss']:8.3f}  "
            f"val {row['val_loss']:8.3f}  val MAE {row['val_mae']:6.3f}"
        )
    print("best epoch:", best_epoch)