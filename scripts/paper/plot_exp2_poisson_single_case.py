"""
从已训练的 exp2 Poisson 2D 实验 run 加载 config 与 checkpoint，对**单个固定算子**绘制论文风格图：

- 3D 面：预测 / 真值 / 误差 / 绝对误差 / 平方误差（一张合成图）
- 2D 伪彩：真值、预测、误差（jet / coolwarm）
- 边际 **Rel L2**（非 MSE / 非 MSRE）：沿 x、沿 y 各一条曲线 — 在每个固定 x（或 y）上，对另一维取整条线，算 \(\|e\|_2/\|p_{\mathrm{true}}\|_2\)

**用法**（在项目根目录）::

    python scripts/paper/plot_exp2_poisson_single_case.py \\
        --run_dir experiments/paper/exp2_poisson_2d/run_YYYYMMDD_HHMMSS \\
        --model_name transformer_deeponet \\
        --checkpoint latest.pt

可选 ``--best_test``：在测试集上按与训练相同的 query 点算 rel_l2，选误差最小的样本再画稠密网格图。

需要 ``config.json`` 与 ``<model_name>/checkpoints/*.pt`` 与训练时一致；数据用相同 ``seed`` 与参数通过 ``generate_poisson_2d_data(..., return_coeffs=True)`` 重现，以得到解析真值。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _load_config(run_dir: Path) -> dict:
    p = run_dir / "config.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _resolve_checkpoint(checkpoints_dir: Path, name: str | None) -> Path:
    if name:
        c = checkpoints_dir / name
        if not c.exists():
            raise FileNotFoundError(f"Checkpoint not found: {c}")
        return c
    latest = checkpoints_dir / "latest.pt"
    if latest.exists():
        return latest
    cpts = sorted(checkpoints_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not cpts:
        raise FileNotFoundError(f"No checkpoints in {checkpoints_dir}")
    return cpts[-1]


def _build_model(cfg: dict, branch: str, bayesian: bool) -> torch.nn.Module:
    num_sensors = cfg["nx"] * cfg["ny"]
    cfg.setdefault("num_sensors", num_sensors)

    from src.models import (
        BayesianDeepONet,
        BayesianFNNBranch,
        BayesianFNNTrunk,
        BayesianTransformerBranch,
        DeepONet,
        FNNBranch,
        FNNTrunk,
        TransformerBranch,
    )

    if branch == "fnn":
        if bayesian:
            branch_m = BayesianFNNBranch(
                cfg["num_sensors"], cfg["branch_hidden"], cfg["output_dim"],
                prior_sigma=cfg["prior_sigma"],
            )
            trunk = BayesianFNNTrunk(
                cfg["coord_dim"], cfg["trunk_hidden"], cfg["output_dim"],
                prior_sigma=cfg["prior_sigma"],
            )
            return BayesianDeepONet(branch_m, trunk, bias=True, min_noise=1e-3)
        b = FNNBranch(cfg["num_sensors"], cfg["branch_hidden"], cfg["output_dim"])
        t = FNNTrunk(cfg["coord_dim"], cfg["trunk_hidden"], cfg["output_dim"])
        return DeepONet(b, t, cfg["output_dim"], bias=True)

    if bayesian:
        branch_m = BayesianTransformerBranch(
            cfg["num_sensors"], cfg["output_dim"],
            d_model=cfg["transformer_d_model"], nhead=cfg["transformer_nhead"],
            num_layers=cfg["transformer_num_layers"], dropout=cfg["transformer_dropout"],
            prior_sigma=cfg["prior_sigma"],
        )
        trunk = BayesianFNNTrunk(
            cfg["coord_dim"], cfg["trunk_hidden"], cfg["output_dim"],
            prior_sigma=cfg["prior_sigma"],
        )
        return BayesianDeepONet(branch_m, trunk, bias=True, min_noise=1e-3)
    b = TransformerBranch(
        cfg["num_sensors"], cfg["output_dim"],
        d_model=cfg["transformer_d_model"], nhead=cfg["transformer_nhead"],
        num_layers=cfg["transformer_num_layers"], dropout=cfg["transformer_dropout"],
    )
    t = FNNTrunk(cfg["coord_dim"], cfg["trunk_hidden"], cfg["output_dim"])
    return DeepONet(b, t, cfg["output_dim"], bias=True)


def _load_weights(model: torch.nn.Module, ckpt_path: Path, device: str) -> None:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob
    model.load_state_dict(sd, strict=True)


def _predict_mean(
    model: torch.nn.Module,
    u: torch.Tensor,
    y: torch.Tensor,
    *,
    bayesian: bool,
    eval_mc_samples: int,
    device: str,
) -> np.ndarray:
    """u: (1, n_sensors), y: (1, n_points, 2) -> (n_points,) numpy float."""
    model.eval()
    with torch.no_grad():
        u = u.to(device)
        y = y.to(device)
        if not bayesian:
            p = model(u, y)
            if p.dim() == 3:
                p = p.squeeze(-1)
            return p.squeeze(0).cpu().numpy()
        preds = []
        for _ in range(eval_mc_samples):
            pred_mean, _, _, _ = model(u, y, sample=True)
            if pred_mean.dim() == 3:
                pred_mean = pred_mean.squeeze(-1)
            preds.append(pred_mean)
        return torch.stack(preds, dim=0).mean(dim=0).squeeze(0).cpu().numpy()


def _rel_l2_on_queries(
    model: torch.nn.Module,
    u: np.ndarray,
    yq: np.ndarray,
    s: np.ndarray,
    *,
    bayesian: bool,
    eval_mc_samples: int,
    device: str,
) -> float:
    u_t = torch.from_numpy(u).float().unsqueeze(0)
    y_t = torch.from_numpy(yq).float().unsqueeze(0)
    s_t = torch.from_numpy(np.asarray(s)).float().reshape(-1)
    pred = torch.from_numpy(
        _predict_mean(model, u_t, y_t, bayesian=bayesian, eval_mc_samples=eval_mc_samples, device=device)
    ).float()
    return (pred - s_t).norm(2).item() / (s_t.norm(2).item() + 1e-8)


def _marginal_rel_l2(err: np.ndarray, p_true: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    err, p_true: meshgrid(..., indexing='xy'), shape (ny, nx).

    For each fixed x (column j): slice Rel L2 = ||e[:,j]||_2 / ||p[:,j]||_2.
    For each fixed y (row i): slice Rel L2 = ||e[i,:]||_2 / ||p[i,:]||_2.

    This matches the usual global Rel L2 definition but restricted to a 1D slice;
    it is not MSRE (mean of squared relative errors per point).
    """
    eps = 1e-10
    nx, ny = err.shape[1], err.shape[0]
    rel_x = np.zeros(nx, dtype=np.float64)
    rel_y = np.zeros(ny, dtype=np.float64)
    for j in range(nx):
        ej, pj = err[:, j], p_true[:, j]
        rel_x[j] = np.linalg.norm(ej) / (np.linalg.norm(pj) + eps)
    for i in range(ny):
        ei, pi = err[i, :], p_true[i, :]
        rel_y[i] = np.linalg.norm(ei) / (np.linalg.norm(pi) + eps)
    x_coords = np.linspace(0.0, 1.0, nx)
    y_coords = np.linspace(0.0, 1.0, ny)
    return x_coords, rel_x, y_coords, rel_y


def run_plot(
    *,
    run_dir: Path,
    model_name: str,
    checkpoint: str | None,
    branch: str,
    bayesian: bool,
    sample_index: int,
    best_test: bool,
    grid_n: int,
    dpi: int,
    eval_mc_samples: int | None,
    out_dir: Path | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = _project_root()
    sys.path.insert(0, str(root))

    from src.data.generators.poisson_2d import fourier_to_solution, generate_poisson_2d_data

    cfg = _load_config(run_dir)
    cfg["num_sensors"] = cfg["nx"] * cfg["ny"]
    if eval_mc_samples is None:
        eval_mc_samples = int(cfg.get("eval_mc_samples", 20))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = run_dir / model_name
    ckpt_path = _resolve_checkpoint(model_dir / "checkpoints", checkpoint)
    model = _build_model(cfg, branch, bayesian)
    _load_weights(model, ckpt_path, device)
    model.to(device)

    data = generate_poisson_2d_data(
        n_train=cfg["n_train"],
        n_test=cfg["n_test"],
        nx=cfg["nx"],
        ny=cfg["ny"],
        n_points_per_sample=cfg["n_points_per_sample"],
        max_mode=cfg["max_mode"],
        length_scale=cfg["length_scale"],
        seed=cfg["seed"],
        query_sampling=cfg["query_sampling"],
        return_coeffs=True,
    )
    coeffs_test = data["coeffs_test"]
    u_test = data["u_test"]
    y_test = data["y_test"]
    s_test = data["s_test"]

    n_test = u_test.shape[0]
    idx = sample_index
    if best_test:
        rels = []
        for i in range(n_test):
            r = _rel_l2_on_queries(
                model,
                u_test[i],
                y_test[i],
                s_test[i],
                bayesian=bayesian,
                eval_mc_samples=eval_mc_samples,
                device=device,
            )
            rels.append(r)
        idx = int(np.argmin(np.array(rels)))
        print(f"[best_test] chose sample index {idx} with rel_l2={rels[idx]:.6f}")

    coeffs = coeffs_test[idx]
    u_one = u_test[idx : idx + 1]

    xs = np.linspace(0.0, 1.0, grid_n, dtype=np.float64)
    ys = np.linspace(0.0, 1.0, grid_n, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    p_true = fourier_to_solution(coeffs, xx.astype(np.float32), yy.astype(np.float32), cfg["max_mode"])

    y_flat = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1).astype(np.float32)
    y_t = torch.from_numpy(y_flat).float().unsqueeze(0)
    u_t = torch.from_numpy(u_one).float()
    p_pred = _predict_mean(
        model, u_t, y_t, bayesian=bayesian, eval_mc_samples=eval_mc_samples, device=device
    ).reshape(grid_n, grid_n)

    err = p_pred - p_true
    abs_err = np.abs(err)
    sq_err = err**2

    out = out_dir or (model_dir / "figures_single_case")
    out.mkdir(parents=True, exist_ok=True)

    # --- 3D: 2 rows — pred, true | error, abs, sq ---
    fig = plt.figure(figsize=(12, 8))
    teal = (0.15, 0.45, 0.55, 0.85)

    def surf(ax, Z, title, zlabel="p"):
        ax.plot_surface(
            xx, yy, Z, color=teal, linewidth=0.2, edgecolor="k", antialiased=True, alpha=0.9
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel(zlabel)
        ax.set_title(title, fontsize=10)

    ax0 = fig.add_subplot(2, 3, 1, projection="3d")
    surf(ax0, p_pred, "pred")
    ax1 = fig.add_subplot(2, 3, 2, projection="3d")
    surf(ax1, p_true, "true")
    fig.add_subplot(2, 3, 3).set_visible(False)
    ax2 = fig.add_subplot(2, 3, 4, projection="3d")
    surf(ax2, err, "error", zlabel="value")
    ax3 = fig.add_subplot(2, 3, 5, projection="3d")
    surf(ax3, abs_err, "abs error", zlabel="value")
    ax4 = fig.add_subplot(2, 3, 6, projection="3d")
    surf(ax4, sq_err, "square error", zlabel="value")
    fig.suptitle(f"单算子可视化 (sample={idx}, ckpt={ckpt_path.name})", fontsize=11)
    plt.tight_layout()
    p3d = out / "single_case_3d_surfaces.png"
    plt.savefig(p3d, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[saved] {p3d}")

    # --- 2D contour ---
    fig2, axs = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, Z, title, cmap in zip(
        axs,
        [p_true, p_pred, err],
        ["p_true", "p_pred", "error"],
        ["jet", "jet", "coolwarm"],
    ):
        tpc = ax.contourf(xx, yy, Z, levels=40, cmap=cmap)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        fig2.colorbar(tpc, ax=ax, shrink=0.85)
    fig2.suptitle(f"Contour (sample={idx})")
    plt.tight_layout()
    p2d = out / "single_case_contourf.png"
    plt.savefig(p2d, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[saved] {p2d}")

    # --- Marginal slice Rel L2 (same definition as global norm ratio, per line) ---
    x_coords, rel_x, y_coords, rel_y = _marginal_rel_l2(err, p_true)
    fig3, axm = plt.subplots(1, 2, figsize=(10, 3.8))
    axm[0].plot(x_coords, rel_x, "b-", lw=1.2)
    axm[0].set_title("Rel L2 vs x (slice along y at each x)")
    axm[0].set_xlabel("x")
    axm[0].set_ylabel(r"Rel L2 ($\|e\|_2/\|p\|_2$ on slice)")
    axm[0].grid(True, alpha=0.3)
    axm[1].plot(y_coords, rel_y, "b-", lw=1.2)
    axm[1].set_title("Rel L2 vs y (slice along x at each y)")
    axm[1].set_xlabel("y")
    axm[1].set_ylabel(r"Rel L2 ($\|e\|_2/\|p\|_2$ on slice)")
    axm[1].grid(True, alpha=0.3)
    fig3.suptitle(
        f"Slice-wise Rel L2 (not MSRE; sample={idx})",
        fontsize=10,
    )
    plt.tight_layout()
    pm = out / "single_case_marginal_rel_l2.png"
    plt.savefig(pm, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[saved] {pm}")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot single-operator figures for exp2 Poisson 2D.")
    p.add_argument("--run_dir", type=Path, required=True, help="Path to run_YYYYMMDD_... (contains config.json)")
    p.add_argument("--model_name", type=str, default="transformer_deeponet", help="Subfolder under run_dir")
    p.add_argument("--checkpoint", type=str, default=None, help="e.g. epoch_200.pt or latest.pt; default: latest or max epoch")
    p.add_argument("--branch", type=str, choices=("transformer", "fnn"), default="transformer")
    p.add_argument("--bayesian", action="store_true", help="Load BayesianDeepONet checkpoint")
    p.add_argument("--sample_index", type=int, default=0, help="Test sample index (ignored if --best_test)")
    p.add_argument("--best_test", action="store_true", help="Pick test sample with smallest rel_l2 on query points")
    p.add_argument("--grid_n", type=int, default=64, help="Dense grid resolution per axis")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--eval_mc_samples", type=int, default=None, help="MC draws for Bayesian mean (default: config)")
    p.add_argument("--out_dir", type=Path, default=None, help="Output directory (default: run_dir/model_name/figures_single_case)")
    args = p.parse_args()
    run_plot(
        run_dir=args.run_dir.resolve(),
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        branch=args.branch,
        bayesian=args.bayesian,
        sample_index=args.sample_index,
        best_test=args.best_test,
        grid_n=args.grid_n,
        dpi=args.dpi,
        eval_mc_samples=args.eval_mc_samples,
        out_dir=args.out_dir.resolve() if args.out_dir else None,
    )


if __name__ == "__main__":
    main()
