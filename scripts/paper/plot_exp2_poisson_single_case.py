"""
从已训练的 exp2 Poisson 2D 实验 run 加载 config 与 checkpoint，对**单个固定算子**绘制论文风格图：

- 3D 面：预测 / 真值 / 误差 / 绝对误差 / 平方误差（一张合成图）
- 2D 伪彩：真值、预测、误差（jet / coolwarm）
- 边际 **Rel L2**（非 MSE / 非 MSRE）：沿 x、沿 y 各一条曲线 — 在每个固定 x（或 y）上，对另一维取整条线，算 \(\|e\|_2/\|p_{\mathrm{true}}\|_2\)

**用法**（在项目根目录）::

    # 默认使用本仓库 run_pi_deeponet_smoke + pi_deeponet + fnn + dpi/grid_n 见 DEFAULT_* 常量
    uv run python scripts/paper/plot_exp2_poisson_single_case.py

    # 指定其它 run
    uv run python scripts/paper/plot_exp2_poisson_single_case.py \\
        --run_dir experiments/paper/exp2_poisson_2d/run_YYYYMMDD_HHMMSS \\
        --model_name pi_deeponet --branch fnn

默认 **不指定 ``--sample_index``** 时，在测试集上按与训练相同的 query 点算 rel_l2，**自动取最小者** 对应的样本。指定 ``--sample_index k`` 则用第 k 条。

需要 ``config.json`` 与 ``<model_name>/checkpoints/*.pt`` 与训练时一致；数据用相同 ``seed`` 与参数通过 ``generate_poisson_2d_data(..., return_coeffs=True)`` 重现，以得到解析真值。

PNG 默认写入仓库根下 ``thesis/figures/``（``DEFAULT_OUT_DIR``）；可用 ``--out_dir`` 覆盖。
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


# 默认与 smoke 实验 ``run_pi_deeponet_smoke`` 一致；无参运行脚本时直接使用
DEFAULT_RUN_DIR = _project_root() / "experiments/paper/exp2_poisson_2d/run_pi_deeponet_smoke"
DEFAULT_MODEL_NAME = "pi_deeponet"
DEFAULT_BRANCH = "fnn"
DEFAULT_GRID_N = 80
DEFAULT_DPI = 200
DEFAULT_OUT_DIR = _project_root() / "thesis" / "figures"


def _configure_matplotlib_chinese_font() -> None:
    """优先使用系统中文字体（Windows 常见为 Microsoft YaHei / SimHei），避免 CJK 显示为方框。"""
    import matplotlib
    from matplotlib import font_manager

    keywords = (
        "YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK",
        "PingFang",
        "Heiti",
        "Song",
        "WenQuanYi",
        "Source Han",
    )
    picked: list[str] = []
    for f in font_manager.fontManager.ttflist:
        name = f.name
        if any(kw in name for kw in keywords):
            picked.append(name)
            break
    if picked:
        matplotlib.rcParams["font.sans-serif"] = picked + ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


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


def _branch_u_panel_text_compact(
    u_flat: np.ndarray,
    coeffs: np.ndarray,
    *,
    idx: int,
    nx: int,
    ny: int,
    max_mode: int,
    picked_by_best: bool,
    query_rel_l2: float | None,
) -> str:
    """第一行第三列信息框用（简短）；论文中完整数值见同目录 single_case_branch_u.txt。"""
    u = np.asarray(u_flat, dtype=np.float64).ravel()
    c = np.asarray(coeffs, dtype=np.float64).ravel()
    n = u.size
    lines = [
        "Branch 输入 u",
        f"f 在 {nx}×{ny} 网格采样，N={n}",
        f"测试索引 k={idx}" + ("（best）" if picked_by_best else ""),
    ]
    if query_rel_l2 is not None:
        lines.append(f"rel_l2@query={query_rel_l2:.5f}")
    lines.append(
        f"u: min={u.min():.4g} max={u.max():.4g} ‖u‖₂={np.linalg.norm(u):.4g}"
    )
    lines.append("Fourier 系数 a:")
    lines.append(np.array2string(c, precision=4, max_line_width=42))
    lines.append("u 前 6 项:")
    lines.append(np.array2string(u[:6], precision=4, max_line_width=42))
    return "\n".join(lines)


def _save_branch_u_file(
    out: Path,
    u_flat: np.ndarray,
    coeffs: np.ndarray,
    *,
    idx: int,
    nx: int,
    ny: int,
    max_mode: int,
    picked_by_best: bool,
    query_rel_l2: float | None,
    cfg_seed: int,
) -> None:
    """论文可引用的完整 u 与元数据（文本）。"""
    u = np.asarray(u_flat, dtype=np.float64).ravel()
    c = np.asarray(coeffs, dtype=np.float64).ravel()
    path = out / "single_case_branch_u.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "# Branch 输入 u：右端项 f 在均匀传感器网格上的离散值（与 generate_poisson_2d_data 中 u_test[k] 一致）\n"
        )
        f.write(f"# nx={nx} ny={ny} N={u.size}  meshgrid indexing=xy 展平顺序与训练数据一致\n")
        f.write(f"# 测试集样本索引 k={idx}  seed={cfg_seed}  best_test={picked_by_best}\n")
        if query_rel_l2 is not None:
            f.write(f"# query 点 rel_l2={query_rel_l2:.18e}\n")
        f.write(f"# Fourier 系数 a (维度 {c.size})，真解由 a 解析给出:\n")
        f.write("# " + " ".join(f"{float(x):.18e}" for x in c) + "\n")
        f.write("# --- u 分量 i, u_i ---\n")
        for i, v in enumerate(u):
            f.write(f"{i}\t{v:.18e}\n")
    print(f"[saved] {path}")


def run_plot(
    *,
    run_dir: Path,
    model_name: str,
    checkpoint: str | None,
    branch: str,
    bayesian: bool,
    sample_index: int | None,
    grid_n: int,
    dpi: int,
    eval_mc_samples: int | None,
    out_dir: Path | None,
    use_cjk_font: bool = True,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    if use_cjk_font:
        _configure_matplotlib_chinese_font()
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
    query_rel_l2: float | None = None
    picked_by_best: bool
    if sample_index is None:
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
        query_rel_l2 = float(rels[idx])
        picked_by_best = True
        print(f"[best_test] chose sample index {idx} with rel_l2={query_rel_l2:.6f}")
    else:
        idx = int(sample_index)
        if idx < 0 or idx >= n_test:
            raise ValueError(f"sample_index={idx} out of range [0, {n_test})")
        picked_by_best = False
        query_rel_l2 = _rel_l2_on_queries(
            model,
            u_test[idx],
            y_test[idx],
            s_test[idx],
            bayesian=bayesian,
            eval_mc_samples=eval_mc_samples,
            device=device,
        )

    coeffs = coeffs_test[idx]
    u_one = u_test[idx : idx + 1]
    u_flat_for_meta = u_test[idx].copy()

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

    out = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    _save_branch_u_file(
        out,
        u_flat_for_meta,
        coeffs,
        idx=idx,
        nx=int(cfg["nx"]),
        ny=int(cfg["ny"]),
        max_mode=int(cfg["max_mode"]),
        picked_by_best=picked_by_best,
        query_rel_l2=query_rel_l2,
        cfg_seed=int(cfg["seed"]),
    )

    panel_txt = _branch_u_panel_text_compact(
        u_flat_for_meta,
        coeffs,
        idx=idx,
        nx=int(cfg["nx"]),
        ny=int(cfg["ny"]),
        max_mode=int(cfg["max_mode"]),
        picked_by_best=picked_by_best,
        query_rel_l2=query_rel_l2,
    )

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
    ax_u = fig.add_subplot(2, 3, 3)
    ax_u.axis("off")
    ax_u.text(
        0.02,
        0.98,
        panel_txt,
        transform=ax_u.transAxes,
        va="top",
        ha="left",
        fontsize=7.5,
    )
    ax2 = fig.add_subplot(2, 3, 4, projection="3d")
    surf(ax2, err, "error", zlabel="value")
    ax3 = fig.add_subplot(2, 3, 5, projection="3d")
    surf(ax3, abs_err, "abs error", zlabel="value")
    ax4 = fig.add_subplot(2, 3, 6, projection="3d")
    surf(ax4, sq_err, "square error", zlabel="value")
    fig.suptitle(f"固定输入u后函数p(x,y)的预测、真值、误差可视化", fontsize=11)
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
    fig2.suptitle(f"真值/预测等高线图和误差热力图")
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
        f"切片 Rel L2（sample={idx}；非 MSRE）",
        fontsize=10,
    )
    plt.tight_layout()
    pm = out / "single_case_marginal_rel_l2.png"
    plt.savefig(pm, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[saved] {pm}")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot single-operator figures for exp2 Poisson 2D.")
    p.add_argument(
        "--run_dir",
        type=Path,
        default=None,
        help=f"Run dir with config.json (default: {DEFAULT_RUN_DIR.name}, from repo root)",
    )
    p.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Subfolder under run_dir")
    p.add_argument("--checkpoint", type=str, default=None, help="e.g. epoch_200.pt or latest.pt; default: latest or max epoch")
    p.add_argument("--branch", type=str, choices=("transformer", "fnn"), default=DEFAULT_BRANCH)
    p.add_argument("--bayesian", action="store_true", help="Load BayesianDeepONet checkpoint")
    p.add_argument(
        "--sample_index",
        type=int,
        default=None,
        help="测试集样本索引；省略则默认按 query 点 rel_l2 最小选取（原 best_test）",
    )
    p.add_argument("--grid_n", type=int, default=DEFAULT_GRID_N, help="Dense grid resolution per axis")
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    p.add_argument("--eval_mc_samples", type=int, default=None, help="MC draws for Bayesian mean (default: config)")
    p.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help=f"Output directory (default: thesis/figures -> {DEFAULT_OUT_DIR})",
    )
    p.add_argument("--no_cjk_font", action="store_true", help="Do not set matplotlib Chinese font (use default DejaVu)")
    args = p.parse_args()
    run_dir = DEFAULT_RUN_DIR if args.run_dir is None else Path(args.run_dir)
    out_plot = args.out_dir.resolve() if args.out_dir is not None else None
    run_plot(
        run_dir=run_dir.resolve(),
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        branch=args.branch,
        bayesian=args.bayesian,
        sample_index=args.sample_index,
        grid_n=args.grid_n,
        dpi=args.dpi,
        eval_mc_samples=args.eval_mc_samples,
        out_dir=out_plot,
        use_cjk_font=not args.no_cjk_font,
    )


if __name__ == "__main__":
    main()
