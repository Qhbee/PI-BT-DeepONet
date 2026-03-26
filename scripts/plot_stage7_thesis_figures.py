"""从 stage7 汇总 CSV 与各 run 目录下的 training_history.json 生成论文用图（thesis/figures）。

数据来源说明
-----------
- **本脚本（当前 thesis 用图）**：读 ``experiments/stage7/<run>/training_history.json``，
  每条记录为 ``epoch, loss, rel_l2, test_mse``（训练结束时由 ``run_stage7_experiments.py`` 写出），
  横轴为 epoch，左图 loss、右图 rel_l2。
- **旧版 ``scripts/plot_paper_figures.py``**：从各 run 目录的 **TensorBoard** 事件里读标量
  ``loss/train``、``metric/rel_l2``（step 与 epoch 对齐方式依赖写入逻辑），图例为英文 ``standard_pi`` 等，
  默认保存到 ``docs/figures/stage7_training_curves*.png``，与 thesis 下路径不同。

运行本脚本默认同时写出曲线长表 CSV（``--export-curves-csv`` 可改路径）。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = _REPO_ROOT / "experiments" / "stage7" / "stage7_summary_epochs15.csv"
DEFAULT_STAGE7_DIR = _REPO_ROOT / "experiments" / "stage7"
DEFAULT_FIG_DIR = _REPO_ROOT / "thesis" / "figures"
DEFAULT_EXPORT_CURVES_CSV = _REPO_ROOT / "experiments" / "stage7" / "stage7_training_curves_long.csv"


def _mode_row_label(mode_key: str) -> str:
    if "hard_bc_pi" in mode_key:
        return "硬边界物理约束"
    if "stabilized_pi" in mode_key:
        return "稳定化物理约束"
    if "standard_pi" in mode_key:
        return "标准物理约束"
    return mode_key


def _load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _setup_cn_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_rel_l2_bar(rows: list[dict[str, str]], out_path: Path) -> None:
    _setup_cn_matplotlib()
    labels = [_mode_row_label(r["mode"]) for r in rows]
    vals = [float(r["rel_l2"]) for r in rows]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(labels, vals, color=colors[: len(vals)], alpha=0.88, edgecolor="0.3", linewidth=0.6)
    ax.set_ylabel("RelL2")
    ax.set_title("扩散反应算例：三种物理约束模式 RelL2（15 epochs，Transformer+Det）")
    ax.grid(True, alpha=0.3, axis="y")
    ymax = max(vals) if vals else 1.0
    # 柱顶数值需要留白；y 上限略高于数据最大值（比例可按需改）
    y_top = ymax * 1.2
    ax.set_ylim(0, y_top)
    label_dy = 0.018 * y_top
    for i, v in enumerate(vals):
        ax.text(i, v + label_dy, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def plot_training_curves(rows: list[dict[str, str]], stage7_dir: Path, out_path: Path) -> None:
    _setup_cn_matplotlib()
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for idx, row in enumerate(rows):
        mode = row["mode"]
        hist_path = stage7_dir / mode / "training_history.json"
        if not hist_path.exists():
            print(f"[warn] skip curves for {mode}: missing {hist_path}")
            continue
        with hist_path.open(encoding="utf-8") as f:
            hist = json.load(f)
        label = _mode_row_label(mode)
        ep = [int(h["epoch"]) for h in hist]
        loss = [float(h["loss"]) for h in hist]
        rel = [float(h.get("rel_l2", 0.0)) for h in hist]
        axes[0].plot(ep, loss, "-", color=colors[idx % len(colors)], alpha=0.9, label=label)
        axes[1].plot(ep, rel, "-", color=colors[idx % len(colors)], alpha=0.9, label=label)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("训练损失")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RelL2")
    axes[1].set_title(r"相对 $L^2$ 误差")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("扩散反应算例：三种物理约束模式（15 epochs）", y=1.02, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def export_curves_long_csv(
    rows: list[dict[str, str]],
    stage7_dir: Path,
    out_csv: Path,
) -> None:
    """将三条曲线的逐 epoch 数据合并为长表 CSV。"""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_key", "label_zh", "epoch", "loss", "rel_l2", "test_mse"])
        for row in rows:
            mode = row["mode"]
            hist_path = stage7_dir / mode / "training_history.json"
            label = _mode_row_label(mode)
            if not hist_path.exists():
                print(f"[warn] export skip {mode}: missing {hist_path}")
                continue
            with hist_path.open(encoding="utf-8") as hf:
                hist = json.load(hf)
            for h in hist:
                w.writerow(
                    [
                        mode,
                        label,
                        int(h["epoch"]),
                        float(h["loss"]),
                        float(h.get("rel_l2", 0.0)),
                        float(h.get("test_mse", 0.0)),
                    ]
                )
    print(f"[saved] {out_csv}")


def main() -> None:
    p = argparse.ArgumentParser(description="由 stage7 CSV + training_history 生成 thesis/figures 插图")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="stage7 汇总 CSV")
    p.add_argument("--stage7-dir", type=Path, default=DEFAULT_STAGE7_DIR, help="experiments/stage7 根目录")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_FIG_DIR, help="论文插图输出目录")
    p.add_argument("--rel-l2-name", type=str, default="stage7_rel_l2.png", help="RelL2 柱状图文件名")
    p.add_argument("--curves-name", type=str, default="stage7_training_curves.png", help="训练曲线图文件名")
    p.add_argument(
        "--export-curves-csv",
        type=Path,
        default=DEFAULT_EXPORT_CURVES_CSV,
        help="训练曲线长表 CSV 输出路径",
    )
    p.add_argument("--no-export-curves-csv", action="store_true", help="不写出曲线 CSV")
    args = p.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV 不存在: {args.csv}")

    rows = _load_csv_rows(args.csv)
    if not rows:
        raise SystemExit(f"CSV 无数据行: {args.csv}")

    plot_rel_l2_bar(rows, args.out_dir / args.rel_l2_name)
    plot_training_curves(rows, args.stage7_dir, args.out_dir / args.curves_name)
    if not args.no_export_curves_csv:
        export_curves_long_csv(rows, args.stage7_dir, args.export_curves_csv)


if __name__ == "__main__":
    main()