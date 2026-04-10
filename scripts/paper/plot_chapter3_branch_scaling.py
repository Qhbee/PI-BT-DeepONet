"""从 thesis 表格导出的 CSV 绘制 FNN Branch vs Transformer Branch 的 scaling 对比图。

每行三个子图：RelL2–参数量、MSE–参数量、训练时间–参数量（时间列 ``time_s``）。

数据来源：``thesis/chap/chapter3.tex`` 表 \\ref{tab:validity-antiderivative} 与
\\ref{tab:validity-poisson2d}，仅保留 Vanilla DeepONet（FNN）与 T-DeepONet（Transformer）。

CSV 默认路径：``thesis/figures/data/chapter3_fn_vs_transformer_scaling.csv``。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CSV = _REPO_ROOT / "thesis" / "figures" / "data" / "chapter3_fn_vs_transformer_scaling.csv"
DEFAULT_OUT = _REPO_ROOT / "thesis" / "figures" / "ch3_branch_scaling_fn_vs_transformer.png"


def _setup_cn_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def load_scaling_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    need = {"task", "task_label", "branch", "params", "time_s", "rel_l2", "mse"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少列: {sorted(missing)}")
    return df


def _params_log_sci_mathtext(x: float, _pos: int) -> str:
    """对数轴刻度值统一为 mathtext：$10^{n}$ 或 $a\\times10^{n}$（避免混用纯整数与科学计数）。"""
    if x <= 0.0 or not np.isfinite(x):
        return ""
    exp = int(np.floor(np.log10(x) + 1e-12))
    coeff = x / (10.0**exp)
    cr = float(np.round(coeff))
    if abs(coeff - cr) < 1e-5:
        c_int = int(cr)
        if c_int == 1:
            return rf"$\mathdefault{{10^{{{exp}}}}}$"
        return rf"$\mathdefault{{{c_int}\times10^{{{exp}}}}}$"
    return rf"$\mathdefault{{{coeff:g}\times10^{{{exp}}}}}$"


def _apply_log_params_xaxis(
    ax,
    param_series: pd.Series,
    *,
    log_subs: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
) -> None:
    """对参数量对数轴：``log_subs`` 为每十倍频上的主刻度系数；标签一律为科学计数 mathtext。"""
    ax.set_xscale("log")
    p = param_series.astype(float)
    lo, hi = float(p.min()), float(p.max())
    ax.set_xlim(lo * 0.8, hi * 1.25)
    nticks = 24 if len(log_subs) > 3 else 15
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=log_subs, numticks=nticks))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs="auto", numticks=12))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_params_log_sci_mathtext))
    ax.tick_params(axis="x", which="major", labelsize=9)


def plot_fn_vs_transformer_scaling(df: pd.DataFrame, out_path: Path) -> None:
    _setup_cn_matplotlib()
    branches = ["FNN", "Transformer"]
    styles = {
        "FNN": {"color": "#2E86AB", "marker": "o", "label": "Vanilla DeepONet（FNN Branch）"},
        "Transformer": {
            "color": "#E94F37",
            "marker": "s",
            "label": "T-DeepONet（Transformer Branch）",
        },
    }
    tasks = df.groupby("task", sort=False).first()["task_label"].to_dict()
    task_keys = list(tasks.keys())

    nrows = len(task_keys)
    fig, axes = plt.subplots(nrows, 3, figsize=(15.0, 4.0 * nrows), sharex=False)
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for row, task in enumerate(task_keys):
        sub = df[df["task"] == task].copy()
        ax_r, ax_m, ax_t = axes[row, 0], axes[row, 1], axes[row, 2]
        for br in branches:
            s = sub[sub["branch"] == br].sort_values("params")
            if s.empty:
                continue
            st = styles[br]
            ax_r.plot(
                s["params"],
                s["rel_l2"],
                color=st["color"],
                marker=st["marker"],
                linewidth=2.0,
                markersize=7,
                label=st["label"],
            )
            mse_plot = s["mse"].astype(float).to_numpy()
            ax_m.plot(
                s["params"],
                mse_plot,
                color=st["color"],
                marker=st["marker"],
                linewidth=2.0,
                markersize=7,
                label=st["label"],
            )
            ax_t.plot(
                s["params"],
                s["time_s"].astype(float),
                color=st["color"],
                marker=st["marker"],
                linewidth=2.0,
                markersize=7,
                label=st["label"],
            )

        params_col = sub["params"]
        # 一维算例跨 10³–10⁴，1–5 主刻度过密；单独用 1–2–5。二维保持 1–5（含 3、4）更细。
        log_subs: tuple[float, ...] = (
            (1.0, 2.0, 5.0) if task == "antiderivative_1d" else (1.0, 2.0, 3.0, 4.0, 5.0)
        )
        _apply_log_params_xaxis(ax_r, params_col, log_subs=log_subs)
        ax_r.set_ylabel("RelL2")
        ax_r.set_title(f"{tasks[task]} — 相对 $L^2$ 误差")
        ax_r.grid(True, alpha=0.35, linestyle="--", which="both")

        _apply_log_params_xaxis(ax_m, params_col, log_subs=log_subs)
        ax_m.set_ylabel("MSE")
        ax_m.set_title(f"{tasks[task]} — MSE")
        ax_m.grid(True, alpha=0.35, linestyle="--", which="both")
        ax_m.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        _apply_log_params_xaxis(ax_t, params_col, log_subs=log_subs)
        ax_t.set_ylabel("时间 (s)")
        ax_t.set_title(f"{tasks[task]} — 训练时间")
        ax_t.grid(True, alpha=0.35, linestyle="--", which="both")

        for ax in (ax_r, ax_m, ax_t):
            ax.set_xlabel("参数量 Params")

        if row == 0:
            ax_r.legend(loc="best", fontsize=8.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="绘制 chapter3 FNN vs Transformer scaling 对比图")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="输入 CSV 路径")
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT, help="输出 PNG 路径")
    args = p.parse_args()
    df = load_scaling_csv(args.csv)
    plot_fn_vs_transformer_scaling(df, args.output)
    print(f"已写出: {args.output}")


if __name__ == "__main__":
    main()
