"""Trainer for DeepONet with TensorBoard."""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm


def train_antiderivative(
    model: nn.Module,
    data: dict,
    lr: float = 0.001,
    epochs: int = 10000,
    batch_size: int = 256,
    log_dir: str | Path = "experiments/antiderivative",
    device: str | None = None,
):
    """Train DeepONet on antiderivative data."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    u_train = torch.from_numpy(data["u_train"])
    y_train = torch.from_numpy(data["y_train"])
    s_train = torch.from_numpy(data["s_train"])
    u_test = torch.from_numpy(data["u_test"])
    y_test = torch.from_numpy(data["y_test"])
    s_test = torch.from_numpy(data["s_test"])

    n_train = u_train.shape[0]
    n_points = y_train.shape[1]
    total_samples = n_train * n_points

    # Flatten for simpler batch sampling
    u_flat = u_train.unsqueeze(1).expand(-1, n_points, -1).reshape(total_samples, -1)
    y_flat = y_train.reshape(total_samples, 1)
    s_flat = s_train.reshape(total_samples)

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)

    pbar = tqdm(range(epochs), desc="Training", ncols=120, miniters=0.1, maxinterval=5)
    for epoch in pbar:
        model.train()
        perm = torch.randperm(total_samples)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, total_samples, batch_size):
            idx = perm[start : start + batch_size]
            u_b = u_flat[idx].to(device)
            y_b = y_flat[idx].to(device)
            s_b = s_flat[idx].to(device)
            pred = model(u_b, y_b)
            loss = mse(pred, s_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        writer.add_scalar("loss/train", avg_loss, epoch)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                y_test_ = y_test.unsqueeze(-1) if y_test.dim() == 2 else y_test
                pred_test = model(u_test.to(device), y_test_.to(device))
                test_mse = mse(pred_test, s_test.to(device)).item()
                s_mean = s_test.float().mean().item()
                rel_l2 = (pred_test.cpu() - s_test).norm(2) / (s_test.norm(2) + 1e-8)
                writer.add_scalar("loss/test_mse", test_mse, epoch)
                writer.add_scalar("metric/rel_l2", rel_l2.item(), epoch)
                pbar.set_postfix(loss=f"{avg_loss:.4f}", rel_l2=f"{rel_l2.item():.4f}")
                print(f"  Epoch {epoch+1}: loss={avg_loss:.6f}, rel_l2={rel_l2.item():.6f}", flush=True)

    writer.close()
    return model
