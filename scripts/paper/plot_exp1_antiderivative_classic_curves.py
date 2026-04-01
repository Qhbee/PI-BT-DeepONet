"""
exp1 反导数：在**解析给定**的输入函数 \(f(x)\) 上构造 Branch 向量 \(u=f(x_{\mathrm{sensors}})\)，
在同一训练好的 DeepONet 上绘制真值反导数 \(s(x)=\int_0^x f(t)\,\mathrm{d}t\)（\(s(0)=0\)）与网络预测 \(\hat s(x)\)，
合成单张 **m×n** 子图，用于展示「同一算子、不同输入」的泛化。

**用法**（在项目根目录）::

    # 无参：使用下方 DEFAULT_RUN_DIR / DEFAULT_MODEL_NAME（与 smoke 训练输出一致）
    uv run python scripts/paper/plot_exp1_antiderivative_classic_curves.py

    uv run python scripts/paper/plot_exp1_antiderivative_classic_curves.py \\
        --run_dir experiments/paper/exp1_baseline_comparison/run_YYYYMMDD_HHMMSS \\
        --model_name pi_bt_deeponet --branch fnn

    # 贝叶斯 checkpoint
    uv run python scripts/paper/plot_exp1_antiderivative_classic_curves.py \\
        --run_dir ... --model_name b_deeponet --branch fnn --bayesian

默认 PNG 写入 ``thesis/figures/exp1_antiderivative_classic_curves.png``（可用 ``--out_dir`` / ``--out_name`` 修改）。
需要 ``config.json`` 与 ``<model_name>/checkpoints/*.pt`` 与训练时一致。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch

# ---------------------------------------------------------------------------
# 默认输出（与 plot_exp2 单样本脚本一致：仓库根下 thesis/figures）
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


DEFAULT_OUT_DIR = _project_root() / "thesis" / "figures"
DEFAULT_OUT_NAME = "exp1_antiderivative_classic_curves.png"

# 与 scripts/paper/_smoke_run_plot_antiderivative.py 一致；无参运行依赖该目录下已训练 checkpoint
DEFAULT_RUN_DIR = _project_root() / "experiments/paper/exp1_antiderivative_smoke"
DEFAULT_MODEL_NAME = "pi_deeponet"

_PLOT_CJK_FONT_FAMILY: str | None = None


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


def _build_model_antiderivative(cfg: dict, branch: str, bayesian: bool) -> torch.nn.Module:
    ns = int(cfg.get("num_sensors", cfg.get("n_sensors")))
    cfg.setdefault("num_sensors", ns)

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
                cfg["num_sensors"],
                cfg["branch_hidden"],
                cfg["output_dim"],
                prior_sigma=cfg["prior_sigma"],
            )
            trunk = BayesianFNNTrunk(
                cfg["coord_dim"],
                cfg["trunk_hidden"],
                cfg["output_dim"],
                prior_sigma=cfg["prior_sigma"],
            )
            return BayesianDeepONet(branch_m, trunk, bias=True, min_noise=1e-3)
        b = FNNBranch(cfg["num_sensors"], cfg["branch_hidden"], cfg["output_dim"])
        t = FNNTrunk(cfg["coord_dim"], cfg["trunk_hidden"], cfg["output_dim"])
        return DeepONet(b, t, cfg["output_dim"], bias=True)

    if bayesian:
        branch_m = BayesianTransformerBranch(
            cfg["num_sensors"],
            cfg["output_dim"],
            d_model=cfg["transformer_d_model"],
            nhead=cfg["transformer_nhead"],
            num_layers=cfg["transformer_num_layers"],
            dropout=cfg["transformer_dropout"],
            prior_sigma=cfg["prior_sigma"],
        )
        trunk = BayesianFNNTrunk(
            cfg["coord_dim"],
            cfg["trunk_hidden"],
            cfg["output_dim"],
            prior_sigma=cfg["prior_sigma"],
        )
        return BayesianDeepONet(branch_m, trunk, bias=True, min_noise=1e-3)
    b = TransformerBranch(
        cfg["num_sensors"],
        cfg["output_dim"],
        d_model=cfg["transformer_d_model"],
        nhead=cfg["transformer_nhead"],
        num_layers=cfg["transformer_num_layers"],
        dropout=cfg["transformer_dropout"],
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
    """u: (1, n_sensors), y: (1, n_points, coord_dim) -> (n_points,) numpy."""
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


def _configure_matplotlib_chinese_font() -> None:
    global _PLOT_CJK_FONT_FAMILY
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
    picked: str | None = None
    for f in font_manager.fontManager.ttflist:
        name = f.name
        if any(kw in name for kw in keywords):
            picked = name
            break
    _PLOT_CJK_FONT_FAMILY = picked
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def _cjk_fontproperties():
    from matplotlib.font_manager import FontProperties

    if _PLOT_CJK_FONT_FAMILY is None:
        return None
    return FontProperties(family=_PLOT_CJK_FONT_FAMILY)


def _domain_from_cfg(cfg: dict) -> tuple[float, float]:
    d = cfg.get("domain")
    if isinstance(d, (list, tuple)) and len(d) >= 2:
        return float(d[0]), float(d[1])
    if isinstance(d, dict) and "min" in d and "max" in d:
        lo = d["min"]
        hi = d["max"]
        if isinstance(lo, list):
            lo = lo[0]
        if isinstance(hi, list):
            hi = hi[0]
        return float(lo), float(hi)
    return 0.0, 1.0


def default_classic_cases() -> (
    list[tuple[str, str, Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]]
):
    """(f 的 mathtext, s 的 mathtext, f(x), s(x)=∫_0^x f). 均在 [0,1] 上数值稳定；s 的式子与 s(0)=0 一致。"""

    def f_one(_x: np.ndarray) -> np.ndarray:
        return np.ones_like(_x, dtype=np.float64)

    def s_one(x: np.ndarray) -> np.ndarray:
        return x

    def f_2x(x: np.ndarray) -> np.ndarray:
        return 2.0 * (x - 0.5)

    def s_2x(x: np.ndarray) -> np.ndarray:
        return x**2 - x

    def f_3x2(x: np.ndarray) -> np.ndarray:
        return 3.0 * (x - 0.5) ** 2

    def s_3x2(x: np.ndarray) -> np.ndarray:
        return (x - 0.5) ** 3 + 0.125

    def f_inv_x1(x: np.ndarray) -> np.ndarray:
        return 1.0 / (x + 1.0)

    def s_inv_x1(x: np.ndarray) -> np.ndarray:
        return np.log(x + 1.0)

    def f_exp(x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    def s_exp(x: np.ndarray) -> np.ndarray:
        return np.exp(x) - 1.0

    def f_ln1p(x: np.ndarray) -> np.ndarray:
        return np.log(1.0 + x)

    def s_ln1p(x: np.ndarray) -> np.ndarray:
        return (1.0 + x) * np.log(1.0 + x) - x

    _tau = 2.0 * np.pi

    def f_sin(x: np.ndarray) -> np.ndarray:
        return np.sin(np.pi * x)

    def s_sin(x: np.ndarray) -> np.ndarray:
        return (1.0 / np.pi) * (1.0 - np.cos(np.pi * x))

    def f_cos(x: np.ndarray) -> np.ndarray:
        return np.cos(2.0 * np.pi * x)

    def s_cos(x: np.ndarray) -> np.ndarray:
        return (0.5 / np.pi) * np.sin(2.0 * np.pi * x)

    def f_neg_inv_sq(x: np.ndarray) -> np.ndarray:
        return -1.0 / (x + 1.0) ** 2

    def s_inv_shift(x: np.ndarray) -> np.ndarray:
        """原函数 1/(x+1)，与 f 配套；配合 run_plot 中减去 s(lo) 得 s(x)=1/(x+1)-1/s(lo+1)，lo=0 时为 1/(x+1)-1。"""
        return 1.0 / (x + 1.0)

    def f_tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def s_tanh(x: np.ndarray) -> np.ndarray:
        return np.log(np.cosh(np.clip(x, -20.0, 20.0)))

    def f_arctan_prime(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + x**2)

    def s_arctan(x: np.ndarray) -> np.ndarray:
        return np.arctan(x)

    def f_sech2(x: np.ndarray) -> np.ndarray:
        c = np.cosh(np.clip(x, -20.0, 20.0))
        return 1.0 / (c * c)

    def s_sech2(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    return [
        (r"$f(x)=1$", r"$s(x)=x$", f_one, s_one),
        (
            r"$f(x)=2(x-\frac{1}{2})$",
            r"$s(x)=x^2-x$",
            f_2x,
            s_2x,
        ),
        (
            r"$f(x)=3(x-\frac{1}{2})^2$",
            r"$s(x)=(x-\frac{1}{2})^3+\frac{1}{8}$",
            f_3x2,
            s_3x2,
        ),
        (r"$f(x)=1/(x+1)$", r"$s(x)=\ln(1+x)$", f_inv_x1, s_inv_x1),
        (r"$f(x)=e^x$", r"$s(x)=e^x-1$", f_exp, s_exp),
        (
            r"$f(x)=\ln(1+x)$",
            r"$s(x)=(1+x)\ln(1+x)-x$",
            f_ln1p,
            s_ln1p,
        ),
        (
            r"$f(x)=\sin(\pi x)$",
            r"$s(x)=\frac{1}{\pi}(1-\cos(\pi x))$",
            f_sin,
            s_sin,
        ),
        (
            r"$f(x)=\cos(2\pi x)$",
            r"$s(x)=\frac{1}{2\pi}\sin(2\pi x)$",
            f_cos,
            s_cos,
        ),
        (
            r"$f(x)=-\frac{1}{(x+1)^2}$",
            r"$s(x)=\frac{1}{x+1}-1$",
            f_neg_inv_sq,
            s_inv_shift,
        ),
        (r"$f(x)=\tanh x$", r"$s(x)=\ln(\cosh x)$", f_tanh, s_tanh),
        (r"$f(x)=1/(1+x^2)$", r"$s(x)=\arctan x$", f_arctan_prime, s_arctan),
        (r"$f(x)=\mathrm{sech}^2 x$", r"$s(x)=\tanh x$", f_sech2, s_sech2),
    ]


def run_plot(
    *,
    run_dir: Path,
    model_name: str,
    checkpoint: str | None,
    branch: str,
    bayesian: bool,
    n_query: int,
    dpi: int,
    eval_mc_samples: int | None,
    out_dir: Path,
    out_name: str,
    nrows: int,
    ncols: int,
    max_panels: int | None,
    use_cjk_font: bool,
    show_legend_panel: int,
) -> None:
    global _PLOT_CJK_FONT_FAMILY

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
    if use_cjk_font:
        _configure_matplotlib_chinese_font()
    else:
        _PLOT_CJK_FONT_FAMILY = None
        matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    fp_cjk = _cjk_fontproperties() if use_cjk_font else None

    import matplotlib.pyplot as plt

    root = _project_root()
    sys.path.insert(0, str(root))

    cfg = _load_config(run_dir)
    if eval_mc_samples is None:
        eval_mc_samples = int(cfg.get("eval_mc_samples", 20))

    lo, hi = _domain_from_cfg(cfg)
    num_sensors = int(cfg.get("num_sensors", cfg.get("n_sensors")))
    x_sensors = np.linspace(lo, hi, num_sensors, dtype=np.float64)
    x_query = np.linspace(lo, hi, n_query, dtype=np.float64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = run_dir / model_name
    ckpt_path = _resolve_checkpoint(model_dir / "checkpoints", checkpoint)
    model = _build_model_antiderivative(cfg, branch, bayesian)
    _load_weights(model, ckpt_path, device)
    model.to(device)

    cases = default_classic_cases()
    if max_panels is not None:
        cases = cases[: max(1, max_panels)]

    n_panels = len(cases)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), squeeze=False)
    flat = axes.ravel()

    y_t = torch.from_numpy(x_query.astype(np.float32)).reshape(1, -1, 1)

    for k, (f_tex, s_tex, f_fn, s_fn) in enumerate(cases):
        ax = flat[k]
        u = f_fn(x_sensors).astype(np.float32)
        u_t = torch.from_numpy(u).unsqueeze(0)
        # s_fn 为 ∫_0^x f；一般区间 [lo,hi] 上真值为 ∫_lo^x f = s_fn(x)-s_fn(lo)
        s_true = s_fn(x_query) - s_fn(np.full_like(x_query, lo, dtype=np.float64))
        s_pred = _predict_mean(
            model, u_t, y_t, bayesian=bayesian, eval_mc_samples=eval_mc_samples, device=device
        )

        ax.plot(x_query, s_true, color="red", lw=1.8, label="true $s(x)$")
        ax.plot(x_query, s_pred, color="blue", lw=1.5, ls="--", label=r"pred $\hat{s}(x)$")
        ax.set_title(f"{f_tex}\n{s_tex}", fontsize=9)
        ax.set_xlabel("$x$", fontsize=9)
        ax.set_ylabel("$s(x)$", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.35)
        if show_legend_panel >= 0 and k == show_legend_panel:
            ax.legend(fontsize=7, loc="best")

    for j in range(n_panels, nrows * ncols):
        flat[j].axis("off")

    supt = r"反导数算子：同一模型下不同输入 $f$ 的 $s(x)=\int_0^x f$ 与预测 $\hat{s}(x)$"
    supt_kw: dict = {"fontsize": 11, "y": 1.02}
    if fp_cjk is not None:
        supt_kw["fontproperties"] = fp_cjk
    fig.suptitle(supt, **supt_kw)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="exp1 反导数：经典 f 上真值与预测曲线合成图")
    p.add_argument(
        "--run_dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=f"含 config.json 的实验目录（默认: {DEFAULT_RUN_DIR}）",
    )
    p.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"模型子目录名（默认: {DEFAULT_MODEL_NAME}，对应 smoke 输出）",
    )
    p.add_argument("--branch", type=str, choices=("fnn", "transformer"), default="fnn")
    p.add_argument("--checkpoint", type=str, default=None, help="默认 latest.pt 或最后 epoch_*.pt")
    p.add_argument("--bayesian", action="store_true", help="加载 BayesianDeepONet checkpoint")
    p.add_argument("--n_query", type=int, default=200, help="查询点个数（稠密折线）")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--eval_mc_samples", type=int, default=None)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--out_name", type=str, default=DEFAULT_OUT_NAME)
    p.add_argument("--nrows", type=int, default=3)
    p.add_argument("--ncols", type=int, default=4)
    p.add_argument("--max_panels", type=int, default=None, help="仅用前 K 个经典函数")
    p.add_argument("--no_cjk_font", action="store_true", help="总标题不用中文字体探测")
    p.add_argument(
        "--legend_panel",
        type=int,
        default=0,
        help="在第几个子图显示图例（0-based），-1 表示不显示",
    )
    args = p.parse_args()

    run_plot(
        run_dir=args.run_dir.resolve(),
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        branch=args.branch,
        bayesian=args.bayesian,
        n_query=args.n_query,
        dpi=args.dpi,
        eval_mc_samples=args.eval_mc_samples,
        out_dir=args.out_dir,
        out_name=args.out_name,
        nrows=args.nrows,
        ncols=args.ncols,
        max_panels=args.max_panels,
        use_cjk_font=not args.no_cjk_font,
        show_legend_panel=args.legend_panel,
    )


if __name__ == "__main__":
    main()
