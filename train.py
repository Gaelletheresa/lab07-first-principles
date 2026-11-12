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


if __name__ == "__main__":
    torch.manual_seed(2026)
    train_loader, _, _ = make_loaders()
    features, targets = next(iter(train_loader))
    small_batch = [(features[:32], targets[:32])]

    model = RentalNet()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for step in range(300):
        loss, mae = run_epoch(model, small_batch, loss_fn, optimizer)
        if step % 50 == 0 or step == 299:
            print(f"step {step:3d}  loss {loss:8.3f}  mae {mae:6.3f}")