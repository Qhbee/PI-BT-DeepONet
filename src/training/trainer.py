"""Trainer for DeepONet with TensorBoard."""

import math

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm

from src.physics import compute_residual


def _alpha_divergence(log_prior: torch.Tensor, log_variational: torch.Tensor, alpha: float) -> torch.Tensor:
    """Compute alpha-divergence term; alpha=1 reduces to KL."""
    if abs(alpha - 1.0) < 1e-8:
        return (log_variational - log_prior).mean()

    log_ratio = (1.0 - alpha) * (log_prior - log_variational)
    log_mean = torch.logsumexp(log_ratio, dim=0) - math.log(log_ratio.numel())
    return log_mean / (alpha * (alpha - 1.0))


def _gaussian_nll(pred_mean: torch.Tensor, pred_std: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dist = torch.distributions.Normal(pred_mean, pred_std)
    return -dist.log_prob(target).mean()


def _antiderivative_anchor_loss(
    model: nn.Module,
    u_batch: torch.Tensor,
    anchor: float,
    bayes_method: str,
    mse: nn.Module,
) -> torch.Tensor:
    """
    Enforce antiderivative anchor condition at x=0 (s(0)=0).
    For this ODE, IC and left-BC are equivalent and use the same anchor.
    """
    y_anchor = torch.full((u_batch.shape[0], 1), anchor, device=u_batch.device, dtype=u_batch.dtype)
    if bayes_method == "alpha_vi":
        pred_anchor, _, _, _ = model(u_batch, y_anchor, sample=False)
    else:
        pred_anchor = model(u_batch, y_anchor)
    return mse(pred_anchor, torch.zeros_like(pred_anchor))


def train_antiderivative(
    model: nn.Module,
    data: dict,
    lr: float = 0.001,
    epochs: int = 10000,
    batch_size: int = 256,
    log_dir: str | Path = "experiments/antiderivative",
    device: str | None = None,
    bayes_method: str = "deterministic",
    uq_mode: str | None = None,
    alpha: float = 1.0,
    mc_samples: int = 3,
    eval_mc_samples: int = 20,
    kl_weight: float | None = None,
    pi_constraint: str = "none",
    pi_weight: float = 0.0,
    bc_weight: float = 0.0,
    ic_weight: float = 0.0,
    n_collocation: int = 256,
):
    """Train DeepONet on antiderivative data."""
    if uq_mode is not None:
        bayes_method = uq_mode

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
    x_sensors = torch.from_numpy(data["x_sensors"]) if "x_sensors" in data else None

    n_train = u_train.shape[0]
    n_points = y_train.shape[1]
    total_samples = n_train * n_points
    num_batches = max(1, math.ceil(total_samples / batch_size))
    effective_kl_weight = kl_weight if kl_weight is not None else (1.0 / num_batches)

    # Flatten for simpler batch sampling
    u_flat = u_train.unsqueeze(1).expand(-1, n_points, -1).reshape(total_samples, -1)
    y_flat = y_train.reshape(total_samples, 1)
    s_flat = s_train.reshape(total_samples)

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)

    use_pde = pi_constraint != "none" and pi_weight > 0 and x_sensors is not None
    use_bc = pi_constraint == "antiderivative" and bc_weight > 0
    use_ic = pi_constraint == "antiderivative" and ic_weight > 0
    if (bc_weight > 0 or ic_weight > 0) and pi_constraint != "antiderivative":
        raise ValueError("Explicit BC/IC losses are currently implemented for pi_constraint='antiderivative' only.")
    domain_min = float(x_sensors[0]) if x_sensors is not None else 0.0
    domain_max = float(x_sensors[-1]) if x_sensors is not None else 1.0

    last_metrics = {"loss": float("nan"), "rel_l2": float("nan"), "test_mse": float("nan")}
    pbar = tqdm(range(epochs), desc="Training", ncols=120, miniters=0.1, maxinterval=5)
    for epoch in pbar:
        model.train()
        perm = torch.randperm(total_samples)
        epoch_loss = 0.0
        epoch_nll = 0.0
        epoch_div = 0.0
        epoch_pde = 0.0
        epoch_bc = 0.0
        epoch_ic = 0.0
        n_batches_epoch = 0
        for start in range(0, total_samples, batch_size):
            idx = perm[start : start + batch_size]
            u_b = u_flat[idx].to(device)
            y_b = y_flat[idx].to(device)
            s_b = s_flat[idx].to(device)

            if bayes_method == "deterministic":
                pred = model(u_b, y_b)
                loss = mse(pred, s_b)
                nll_for_log = loss.detach()
                div_for_log = torch.zeros_like(nll_for_log)
            elif bayes_method == "alpha_vi":
                mc_nll = []
                mc_log_prior = []
                mc_log_variational = []
                for _ in range(mc_samples):
                    pred_mean, pred_std, log_prior, log_variational = model(u_b, y_b, sample=True)
                    mc_nll.append(_gaussian_nll(pred_mean, pred_std, s_b))
                    mc_log_prior.append(log_prior)
                    mc_log_variational.append(log_variational)

                nll = torch.stack(mc_nll).mean()
                log_prior_tensor = torch.stack(mc_log_prior)
                log_variational_tensor = torch.stack(mc_log_variational)
                divergence = _alpha_divergence(log_prior_tensor, log_variational_tensor, alpha)
                loss = nll + effective_kl_weight * divergence
                nll_for_log = nll.detach()
                div_for_log = divergence.detach()
            else:
                raise ValueError(f"Unsupported bayes_method: {bayes_method}")

            if use_pde:
                batch_size_cur = u_b.shape[0]
                y_colloc = (
                    torch.rand(batch_size_cur, n_collocation, 1, device=device, dtype=u_b.dtype)
                    * (domain_max - domain_min)
                    + domain_min
                )
                x_sensors_dev = x_sensors.to(device)
                L_pde = compute_residual(
                    model, u_b, y_colloc, x_sensors_dev,
                    pde_type=pi_constraint,
                    sample=(bayes_method == "alpha_vi"),
                )
                loss = loss + pi_weight * L_pde
                epoch_pde += L_pde.detach().item()

            if use_bc or use_ic:
                # Antiderivative anchor condition at x=0.
                L_anchor = _antiderivative_anchor_loss(
                    model=model,
                    u_batch=u_b,
                    anchor=domain_min,
                    bayes_method=bayes_method,
                    mse=mse,
                )
                if use_bc:
                    loss = loss + bc_weight * L_anchor
                    epoch_bc += L_anchor.detach().item()
                if use_ic:
                    loss = loss + ic_weight * L_anchor
                    epoch_ic += L_anchor.detach().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_nll += nll_for_log.item()
            epoch_div += div_for_log.item()
            n_batches_epoch += 1

        avg_loss = epoch_loss / n_batches_epoch
        avg_nll = epoch_nll / n_batches_epoch
        avg_div = epoch_div / n_batches_epoch
        avg_pde = epoch_pde / n_batches_epoch if use_pde else 0.0
        avg_bc = epoch_bc / n_batches_epoch if use_bc else 0.0
        avg_ic = epoch_ic / n_batches_epoch if use_ic else 0.0
        writer.add_scalar("loss/train", avg_loss, epoch)
        writer.add_scalar("loss/train_nll", avg_nll, epoch)
        writer.add_scalar("loss/train_divergence", avg_div, epoch)
        writer.add_scalar("loss/pde", avg_pde, epoch)
        writer.add_scalar("loss/bc", avg_bc, epoch)
        writer.add_scalar("loss/ic", avg_ic, epoch)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                y_test_ = y_test.unsqueeze(-1) if y_test.dim() == 2 else y_test
                if bayes_method == "deterministic":
                    pred_test = model(u_test.to(device), y_test_.to(device))
                    test_mse = mse(pred_test, s_test.to(device)).item()
                    rel_l2 = (pred_test.cpu() - s_test).norm(2) / (s_test.norm(2) + 1e-8)
                else:
                    pred_samples = []
                    aleatoric_var_samples = []
                    for _ in range(eval_mc_samples):
                        pred_mean, pred_std, _, _ = model(u_test.to(device), y_test_.to(device), sample=True)
                        pred_samples.append(pred_mean)
                        aleatoric_var_samples.append(pred_std**2)

                    pred_stack = torch.stack(pred_samples, dim=0)
                    pred_test = pred_stack.mean(dim=0)
                    epistemic_std = pred_stack.std(dim=0)
                    aleatoric_std = torch.stack(aleatoric_var_samples, dim=0).mean(dim=0).sqrt()
                    total_std = torch.sqrt(epistemic_std**2 + aleatoric_std**2)
                    writer.add_scalar("metric/epistemic_std_mean", epistemic_std.mean().item(), epoch)
                    writer.add_scalar("metric/aleatoric_std_mean", aleatoric_std.mean().item(), epoch)
                    writer.add_scalar("metric/total_std_mean", total_std.mean().item(), epoch)
                    test_mse = mse(pred_test, s_test.to(device)).item()
                    rel_l2 = (pred_test.cpu() - s_test).norm(2) / (s_test.norm(2) + 1e-8)

                writer.add_scalar("loss/test_mse", test_mse, epoch)
                writer.add_scalar("metric/rel_l2", rel_l2.item(), epoch)
                last_metrics["rel_l2"] = rel_l2.item()
                last_metrics["test_mse"] = test_mse
                pbar.set_postfix(loss=f"{avg_loss:.4f}", rel_l2=f"{rel_l2.item():.4f}")
                print(f"  Epoch {epoch+1}: loss={avg_loss:.6f}, rel_l2={rel_l2.item():.6f}", flush=True)
        last_metrics["loss"] = avg_loss

    writer.close()
    return model, last_metrics
