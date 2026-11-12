import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn

from model import RentalNet
from prepare import make_loaders

REPORTS = Path("reports")
SEED = 2026


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


def arrays_from(loader):
    """Get the prepared features and targets back out of a loader as arrays."""
    return loader.dataset.features.numpy(), loader.dataset.targets.numpy()


def baseline_metrics(train_loader, val_loader, test_loader):
    """Score the training-mean baseline and linear regression (MAE and MSE)."""
    x_train, y_train = arrays_from(train_loader)
    mean_value = float(y_train.mean())
    linear = LinearRegression().fit(x_train, y_train)

    results = {"train_mean": {}, "linear": {}}
    for split, loader in [("val", val_loader), ("test", test_loader)]:
        x, y = arrays_from(loader)
        predictions = {
            "train_mean": np.full_like(y, mean_value),
            "linear": linear.predict(x),
        }
        for name, predicted in predictions.items():
            results[name][f"{split}_mae"] = float(mean_absolute_error(y, predicted))
            results[name][f"{split}_mse"] = float(mean_squared_error(y, predicted))
    return results


def plot_curves(history, best_epoch, path, title):
    """Plot training and validation loss for every epoch, and mark the best."""
    frame = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(frame["epoch"], frame["train_loss"], label="training loss")
    ax.plot(frame["epoch"], frame["val_loss"], label="validation loss")
    ax.axvline(best_epoch, color="gray", linestyle="--", label=f"best epoch {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    torch.manual_seed(SEED)
    REPORTS.mkdir(exist_ok=True)
    train_loader, val_loader, test_loader = make_loaders()

    baselines = baseline_metrics(train_loader, val_loader, test_loader)
    for name, scores in baselines.items():
        print(
            f"{name:11s} val MAE {scores['val_mae']:6.3f}  "
            f"val MSE {scores['val_mse']:8.3f}"
        )

    model = RentalNet(hidden=64)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    history, best_epoch = train_model(
        model, train_loader, val_loader, loss_fn, optimizer, epochs=200
    )
    for row in history:
        if row["epoch"] % 25 == 0:
            print(
                f"epoch {row['epoch']:3d}  train {row['train_loss']:8.3f}  "
                f"val {row['val_loss']:8.3f}  val MAE {row['val_mae']:6.3f}"
            )
    print("best epoch:", best_epoch)
    print("best val MAE:", round(history[best_epoch - 1]["val_mae"], 3))

    pd.DataFrame(history).to_csv(REPORTS / "history.csv", index=False)
    plot_curves(
        history, best_epoch, REPORTS / "learning_curves.png", "Main run (hidden 64)"
    )