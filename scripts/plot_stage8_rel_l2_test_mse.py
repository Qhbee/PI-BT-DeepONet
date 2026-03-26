"""从 Stage8 汇总 CSV 或内置表格数据绘制 RelL2 / TestMSE 分组柱状图。

左 y 轴为 RelL2（蓝绿 teal），右 y 轴为 TestMSE（浅粉），量纲分开更易读。
与 ``run_stage8_experiments.py`` 输出的 ``stage8_summary_*.csv`` 列名兼容；
未提供 CSV 时使用论文表格中的默认数值。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# 默认：与论文表一致（Trunk / RelL2 / TestMSE）
DEFAULT_ROWS: list[tuple[str, float, float]] = [
    ("FNN", 0.0758, 0.0149),
    ("POD", 0.0704, 0.0137),
    ("Ex", 0.0516, 0.0102),
    ("ExV2", 0.0493, 0.0097),
]

# 与论文原图 Stage8 RelL2 单柱图一致的蓝绿色（teal）
COLOR_REL_L2 = "#3d9494"
COLOR_REL_L2_EDGE = "#2a6f6f"
COLOR_TEST_MSE = "#F8BBD0"
# 与填充同色相、略加深（类比 teal 填+深青描边），避免 #E91E63 那种高对比“描边感”
COLOR_TEST_MSE_EDGE = "#c896aa"
COLOR_TEST_MSE_AXIS = "#6b4d5c"  # 右侧轴标题/刻度：可读、不与柱体抢对比


def _mode_to_label(mode: str) -> str:
    m = mode.strip().lower()
    if m.startswith("trunk_"):
        m = m[len("trunk_") :]
    for suf in ("_transformer_bayes",):
        if m.endswith(suf):
            m = m[: -len(suf)]
    mapping = {"fnn": "FNN", "pod": "POD", "ex": "Ex", "ex_v2": "ExV2"}
    return mapping.get(m, mode)


def load_csv(path: Path) -> list[tuple[str, float, float]]:
    rows: list[tuple[str, float, float]] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rows
        # 统一小写键
        for raw in reader:
            row = {k.strip().lower(): v for k, v in raw.items()}
            mode = row.get("mode") or row.get("trunk") or row.get("name")
            rel = row.get("rel_l2")
            mse = row.get("test_mse")
            if mode is None or rel is None or mse is None:
                continue
            rows.append((_mode_to_label(str(mode)), float(rel), float(mse)))
    return rows


def plot_grouped_bars(
    labels: list[str],
    rel_l2: np.ndarray,
    test_mse: np.ndarray,
    out_path: Path,
    dpi: int = 150,
) -> None:
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(
        x - width / 2,
        rel_l2,
        width,
        label="RelL2",
        color=COLOR_REL_L2,
        edgecolor=COLOR_REL_L2_EDGE,
        linewidth=0.6,
        zorder=2,
    )

    ax.set_ylim(0.0, 0.1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RelL2", color=COLOR_REL_L2_EDGE)
    ax.tick_params(axis="y", labelcolor=COLOR_REL_L2_EDGE)
    ax.set_title("Stage 8: Trunk extension (RelL2 & TestMSE)")
    ax.grid(True, axis="y", alpha=0.35, linestyle="--", zorder=0)

    ax2 = ax.twinx()
    ax2.bar(
        x + width / 2,
        test_mse,
        width,
        label="TestMSE",
        color=COLOR_TEST_MSE,
        edgecolor=COLOR_TEST_MSE_EDGE,
        linewidth=0.5,
        alpha=0.95,
        zorder=2,
    )
    ax2.set_ylim(0.0, 0.05)
    ax2.set_ylabel("TestMSE", color=COLOR_TEST_MSE_EDGE)
    ax2.tick_params(axis="y", labelcolor=COLOR_TEST_MSE_EDGE)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", framealpha=0.92)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Plot Stage8 RelL2 & TestMSE bar chart.")
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="stage8_summary_*.csv（列含 mode, rel_l2, test_mse）；缺省用内置表",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=ROOT / "docs" / "figures" / "stage8_rel_l2_test_mse.png",
        help="输出 PNG 路径",
    )
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()
    if args.csv and args.csv.exists():
        data = load_csv(args.csv)
        if not data:
            raise SystemExit(f"[错误] CSV 无有效行: {args.csv}")
    else:
        if args.csv:
            print(f"[提示] 未找到 {args.csv}，使用内置默认数据。")
        data = DEFAULT_ROWS

    labels = [d[0] for d in data]
    rel_l2 = np.array([d[1] for d in data], dtype=float)
    test_mse = np.array([d[2] for d in data], dtype=float)

    plot_grouped_bars(labels, rel_l2, test_mse, args.out, dpi=args.dpi)
    print(f"[SAVED] {args.out.resolve()}")


if __name__ == "__main__":
    main()
