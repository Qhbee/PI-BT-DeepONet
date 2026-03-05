"""Trainer for DeepONet with TensorBoard."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
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
    return -torch.distributions.Normal(pred_mean, pred_std).log_prob(target).mean()


def _predict(model: nn.Module, u: torch.Tensor, y: torch.Tensor, bayes_method: str, sample: bool) -> torch.Tensor:
    if bayes_method == "alpha_vi":
        pred, _, _, _ = model(u, y, sample=sample)
        return pred
    return model(u, y)


def _interp1d_batched(x_sensors: torch.Tensor, u: torch.Tensor, x_query: torch.Tensor) -> torch.Tensor:
    """Batched 1D interpolation over x_sensors, per sample."""
    if x_query.dim() == 3:
        x_query = x_query.squeeze(-1)
    m = x_sensors.shape[0]
    idx_right = torch.searchsorted(x_sensors, x_query, right=False)
    idx_right = torch.clamp(idx_right, 1, m - 1)
    idx_left = idx_right - 1
    x_left = x_sensors[idx_left]
    x_right = x_sensors[idx_right]
    u_left = u.gather(1, idx_left)
    u_right = u.gather(1, idx_right)
    w = (x_query - x_left) / (x_right - x_left + 1e-8)
    return u_left + w * (u_right - u_left)


def _compute_ic_bc_losses(
    case: str,
    model: nn.Module,
    u_batch: torch.Tensor,
    x_sensors: torch.Tensor | None,
    domain_min: torch.Tensor,
    domain_max: torch.Tensor,
    mse: nn.Module,
    bayes_method: str,
    n_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (L_bc, L_ic) for selected case."""
    device = u_batch.device
    dtype = u_batch.dtype
    batch = u_batch.shape[0]
    zero = torch.zeros((), device=device, dtype=dtype)

    if case == "antiderivative":
        y_anchor = torch.full((batch, 1), float(domain_min[0].item()), device=device, dtype=dtype)
        pred = _predict(model, u_batch, y_anchor, bayes_method=bayes_method, sample=False)
        anchor_loss = mse(pred, torch.zeros_like(pred))
        return anchor_loss, anchor_loss

    if case == "burgers":
        # IC: s(x,0) = u(x)
        x_ic = torch.rand(batch, n_samples, 1, device=device, dtype=dtype) * (domain_max[1] - domain_min[1]) + domain_min[1]
        t0 = torch.zeros(batch, n_samples, 1, device=device, dtype=dtype)
        y_ic = torch.cat([t0, x_ic], dim=-1)
        pred_ic = _predict(model, u_batch, y_ic, bayes_method=bayes_method, sample=False)
        if x_sensors is None:
            raise ValueError("x_sensors is required for Burgers IC interpolation.")
        target_ic = _interp1d_batched(x_sensors, u_batch, x_ic.squeeze(-1))
        L_ic = mse(pred_ic, target_ic)

        # Periodic BC: s(t,0)=s(t,1), s_x(t,0)=s_x(t,1)
        t_bc = torch.rand(batch, n_samples, 1, device=device, dtype=dtype) * (domain_max[0] - domain_min[0]) + domain_min[0]
        x0 = torch.full_like(t_bc, domain_min[1])
        x1 = torch.full_like(t_bc, domain_max[1])
        y0 = torch.cat([t_bc, x0], dim=-1).detach().clone().requires_grad_(True)
        y1 = torch.cat([t_bc, x1], dim=-1).detach().clone().requires_grad_(True)
        p0 = _predict(model, u_batch, y0, bayes_method=bayes_method, sample=False)
        p1 = _predict(model, u_batch, y1, bayes_method=bayes_method, sample=False)
        g0 = torch.autograd.grad(outputs=p0.sum(), inputs=y0, create_graph=True)[0][..., 1]
        g1 = torch.autograd.grad(outputs=p1.sum(), inputs=y1, create_graph=True)[0][..., 1]
        L_bc = mse(p0, p1) + mse(g0, g1)
        return L_bc, L_ic

    if case == "diffusion_reaction":
        # IC at t=0 and BC at x=0,1 are zero.
        x_ic = torch.rand(batch, n_samples, 1, device=device, dtype=dtype) * (domain_max[1] - domain_min[1]) + domain_min[1]
        t0 = torch.zeros(batch, n_samples, 1, device=device, dtype=dtype)
        y_ic = torch.cat([t0, x_ic], dim=-1)
        p_ic = _predict(model, u_batch, y_ic, bayes_method=bayes_method, sample=False)
        L_ic = mse(p_ic, torch.zeros_like(p_ic))

        t_bc = torch.rand(batch, n_samples, 1, device=device, dtype=dtype) * (domain_max[0] - domain_min[0]) + domain_min[0]
        y_l = torch.cat([t_bc, torch.full_like(t_bc, domain_min[1])], dim=-1)
        y_r = torch.cat([t_bc, torch.full_like(t_bc, domain_max[1])], dim=-1)
        p_l = _predict(model, u_batch, y_l, bayes_method=bayes_method, sample=False)
        p_r = _predict(model, u_batch, y_r, bayes_method=bayes_method, sample=False)
        L_bc = mse(p_l, torch.zeros_like(p_l)) + mse(p_r, torch.zeros_like(p_r))
        return L_bc, L_ic

    if case == "darcy":
        n_edge = n_samples
        s = torch.rand(batch, n_edge, 1, device=device, dtype=dtype)
        x0 = torch.zeros_like(s)
        x1 = torch.ones_like(s)
        y_btm = torch.cat([s, x0], dim=-1)
        y_top = torch.cat([s, x1], dim=-1)
        y_lft = torch.cat([x0, s], dim=-1)
        y_rgt = torch.cat([x1, s], dim=-1)
        p_b = _predict(model, u_batch, y_btm, bayes_method=bayes_method, sample=False)
        p_t = _predict(model, u_batch, y_top, bayes_method=bayes_method, sample=False)
        p_l = _predict(model, u_batch, y_lft, bayes_method=bayes_method, sample=False)
        p_r = _predict(model, u_batch, y_rgt, bayes_method=bayes_method, sample=False)
        L_bc = mse(p_b, torch.zeros_like(p_b)) + mse(p_t, torch.zeros_like(p_t)) + mse(p_l, torch.zeros_like(p_l)) + mse(p_r, torch.zeros_like(p_r))
        return L_bc, zero

    return zero, zero


def train_operator(
    model: nn.Module,
    data: dict,
    case: str = "antiderivative",
    lr: float = 0.001,
    epochs: int = 10000,
    batch_size: int = 256,
    log_dir: str | Path = "experiments",
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
    diffusion_D: float = 0.01,
    reaction_k: float = 0.1,
    burgers_nu: float = 0.01 / math.pi,
):
    """Generic trainer for single-output operator tasks."""
    if uq_mode is not None:
        bayes_method = uq_mode

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    u_train = torch.from_numpy(data["u_train"]).float()
    y_train = torch.from_numpy(data["y_train"]).float()
    s_train = torch.from_numpy(data["s_train"]).float()
    u_test = torch.from_numpy(data["u_test"]).float()
    y_test = torch.from_numpy(data["y_test"]).float()
    s_test = torch.from_numpy(data["s_test"]).float()
    x_sensors = torch.from_numpy(data["x_sensors"]).float() if "x_sensors" in data else None

    if y_train.dim() == 2:
        y_train = y_train.unsqueeze(-1)
    if y_test.dim() == 2:
        y_test = y_test.unsqueeze(-1)

    coord_dim = y_train.shape[-1]
    n_train = u_train.shape[0]
    n_points = y_train.shape[1]
    total_samples = n_train * n_points
    num_batches = max(1, math.ceil(total_samples / batch_size))
    effective_kl_weight = kl_weight if kl_weight is not None else (1.0 / num_batches)

    u_flat = u_train.unsqueeze(1).expand(-1, n_points, -1).reshape(total_samples, -1)
    y_flat = y_train.reshape(total_samples, coord_dim)
    s_flat = s_train.reshape(total_samples)

    if "domain" in data:
        domain_min = torch.tensor(data["domain"]["min"], dtype=torch.float32, device=device)
        domain_max = torch.tensor(data["domain"]["max"], dtype=torch.float32, device=device)
    elif x_sensors is not None:
        domain_min = torch.tensor([float(x_sensors[0])], dtype=torch.float32, device=device)
        domain_max = torch.tensor([float(x_sensors[-1])], dtype=torch.float32, device=device)
    else:
        domain_min = torch.zeros(coord_dim, dtype=torch.float32, device=device)
        domain_max = torch.ones(coord_dim, dtype=torch.float32, device=device)

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)

    use_pde = pi_constraint != "none" and pi_weight > 0
    use_bc = bc_weight > 0
    use_ic = ic_weight > 0

    last_metrics = {"loss": float("nan"), "rel_l2": float("nan"), "test_mse": float("nan")}
    pbar = tqdm(range(epochs), desc=f"Training[{case}]", ncols=120, miniters=0.1, maxinterval=5)

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
                low = domain_min.view(1, 1, -1)
                span = (domain_max - domain_min).view(1, 1, -1)
                y_colloc = torch.rand(batch_size_cur, n_collocation, coord_dim, device=device, dtype=u_b.dtype) * span + low
                x_sensors_dev = x_sensors.to(device) if x_sensors is not None else torch.linspace(0.0, 1.0, u_b.shape[1], device=device, dtype=u_b.dtype)
                L_pde = compute_residual(
                    model=model,
                    u=u_b,
                    y=y_colloc,
                    x_sensors=x_sensors_dev,
                    pde_type=pi_constraint,
                    sample=(bayes_method == "alpha_vi"),
                    nu=burgers_nu,
                    diffusion_D=diffusion_D,
                    reaction_k=reaction_k,
                )
                loss = loss + pi_weight * L_pde
                epoch_pde += L_pde.detach().item()

            if use_bc or use_ic:
                x_sensors_dev = x_sensors.to(device) if x_sensors is not None else None
                L_bc, L_ic = _compute_ic_bc_losses(
                    case=case,
                    model=model,
                    u_batch=u_b,
                    x_sensors=x_sensors_dev,
                    domain_min=domain_min,
                    domain_max=domain_max,
                    mse=mse,
                    bayes_method=bayes_method,
                    n_samples=max(8, n_collocation // 8),
                )
                if use_bc:
                    loss = loss + bc_weight * L_bc
                    epoch_bc += L_bc.detach().item()
                if use_ic:
                    loss = loss + ic_weight * L_ic
                    epoch_ic += L_ic.detach().item()

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
                y_test_dev = y_test.to(device)
                u_test_dev = u_test.to(device)
                s_test_dev = s_test.to(device)
                if bayes_method == "deterministic":
                    pred_test = model(u_test_dev, y_test_dev)
                    test_mse = mse(pred_test, s_test_dev).item()
                    rel_l2 = (pred_test.cpu() - s_test).norm(2) / (s_test.norm(2) + 1e-8)
                else:
                    pred_samples = []
                    aleatoric_var_samples = []
                    for _ in range(eval_mc_samples):
                        pred_mean, pred_std, _, _ = model(u_test_dev, y_test_dev, sample=True)
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
                    test_mse = mse(pred_test, s_test_dev).item()
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
    """Backward-compatible antiderivative trainer wrapper."""
    return train_operator(
        model=model,
        data=data,
        case="antiderivative",
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        log_dir=log_dir,
        device=device,
        bayes_method=bayes_method,
        uq_mode=uq_mode,
        alpha=alpha,
        mc_samples=mc_samples,
        eval_mc_samples=eval_mc_samples,
        kl_weight=kl_weight,
        pi_constraint=pi_constraint,
        pi_weight=pi_weight,
        bc_weight=bc_weight,
        ic_weight=ic_weight,
        n_collocation=n_collocation,
    )
