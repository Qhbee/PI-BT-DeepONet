"""绘制 stage6 下 8 个 ns_* 实验的 training_history 曲线（2×3 子图，不做插值）。

布局：
- 第 1 行 Kovasznay：Loss | RelL2 | Test MSE，每个子图 4 条线（参数化/BC→场 × Hard/CLS）。
- 第 2 行 Beltrami：同上，分支 B 为 IC→场。

数据：各目录 ``experiments/stage6/<run>/training_history.json``（epoch, loss, rel_l2, test_mse）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STAGE6_DIR = _REPO_ROOT / "experiments" / "stage6"
DEFAULT_OUT = _REPO_ROOT / "thesis" / "figures" / "stage6_ns_training_curves_2x3.png"

# (文件夹名, 图例名) — 顺序固定，与颜色一一对应
KOVASZNAY_RUNS: list[tuple[str, str]] = [
    ("ns_kovasznay_parametric_hard", "参数化 Hard"),
    ("ns_kovasznay_parametric_cls", "参数化 CLS"),
    ("ns_kovasznay_bc2field_hard", "BC→场 Hard"),
    ("ns_kovasznay_bc2field_cls", "BC→场 CLS"),
]

BELTRAMI_RUNS: list[tuple[str, str]] = [
    ("ns_beltrami_parametric_hard", "参数化 Hard"),
    ("ns_beltrami_parametric_cls", "参数化 CLS"),
    ("ns_beltrami_ic2field_hard", "IC→场 Hard"),
    ("ns_beltrami_ic2field_cls", "IC→场 CLS"),
]

COLORS = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]

METRICS: list[tuple[str, str, str]] = [
    ("loss", "Loss", "训练损失"),
    ("rel_l2", "RelL2", r"相对 $L^2$ 误差"),
    ("test_mse", "Test MSE", "Test MSE"),
]


def _setup_cn_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _load_history(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _plot_metric_panel(
    ax: plt.Axes,
    stage6_dir: Path,
    runs: list[tuple[str, str]],
    metric_key: str,
    ylabel: str,
    title: str,
    *,
    show_legend: bool,
) -> None:
    for idx, (folder, label) in enumerate(runs):
        hist_path = stage6_dir / folder / "training_history.json"
        if not hist_path.exists():
            print(f"[warn] 缺失 {hist_path}")
            continue
        hist = _load_history(hist_path)
        ep = [int(h["epoch"]) for h in hist]
        y = [float(h.get(metric_key, 0.0)) for h in hist]
        ax.plot(ep, y, "-", color=COLORS[idx % len(COLORS)], alpha=0.9, label=label, linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(fontsize=7, loc="best", framealpha=0.92)


def plot_ns_grid(stage6_dir: Path, out_path: Path) -> None:
    _setup_cn_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex="col")

    for col, (mkey, ylab, _mtitle) in enumerate(METRICS):
        _plot_metric_panel(
            axes[0, col],
            stage6_dir,
            KOVASZNAY_RUNS,
            mkey,
            ylab,
            f"Kovasznay：{_mtitle}",
            show_legend=(col == 2),
        )
        _plot_metric_panel(
            axes[1, col],
            stage6_dir,
            BELTRAMI_RUNS,
            mkey,
            ylab,
            f"Beltrami：{_mtitle}",
            show_legend=(col == 2),
        )

    fig.suptitle("Navier-Stokes（stage6）：Kovasznay / Beltrami 训练曲线", fontsize=12, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage6-dir", type=Path, default=DEFAULT_STAGE6_DIR, help="experiments/stage6 根目录")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出 PNG 路径")
    args = p.parse_args()

    if not args.stage6_dir.is_dir():
        raise SystemExit(f"目录不存在: {args.stage6_dir}")

    plot_ns_grid(args.stage6_dir, args.out)


if __name__ == "__main__":
    main()
