from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_CSV = _SCRIPT_DIR / "experiments" / "factors_data.csv"


def _parse_combo(combo: str) -> tuple[str, str, str]:
    """从「组合」列解析 贝叶斯方法、架构、物理约束（有/无）。"""
    parts = [p.strip() for p in combo.split("+")]
    if len(parts) != 3:
        raise ValueError(f"组合列需为「A + B + C」三段，当前: {combo!r}")
    bayes, arch, phys_full = parts
    if "有物理" in phys_full:
        phys = "有"
    elif "无物理" in phys_full or phys_full.startswith("无"):
        phys = "无"
    else:
        raise ValueError(f"无法识别物理约束子串: {phys_full!r}（组合: {combo!r}）")
    return bayes, arch, phys


def load_factors_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "组合" not in df.columns:
        raise ValueError(f"CSV 缺少「组合」列: {csv_path}")
    parsed = df["组合"].map(lambda c: _parse_combo(str(c)))
    df = df.copy()
    df["贝叶斯方法"] = [p[0] for p in parsed]
    df["架构"] = [p[1] for p in parsed]
    df["物理约束"] = [p[2] for p in parsed]
    return df


def _resolve_cjk_sans_family() -> str:
    """本机第一个可用的无衬线 CJK 字体名（用于与 DejaVu 组成回退链）。"""
    for name in (
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "SimHei",
        "STSong",
    ):
        try:
            fm.findfont(FontProperties(family=name), fallback_to_default=False)
        except ValueError:
            continue
        return name
    raise RuntimeError(
        "未找到常见 CJK 无衬线字体（如 Microsoft YaHei）。请安装字体或通过环境配置字体路径。"
    )


def _plot_fonts() -> tuple[FontProperties, FontProperties]:
    """(纯拉丁/数字用 DejaVu, 含中文的字符串用 DejaVu→CJK 回退)."""
    fp_dejavu = FontProperties(family="DejaVu Sans")
    cjk = _resolve_cjk_sans_family()
    fp_mixed = FontProperties(family=["DejaVu Sans", cjk])
    return fp_dejavu, fp_mixed


def _mse_four_bar_profile(df: pd.DataFrame, bayes: str) -> np.ndarray:
    """按 FNN无 → FNN有 → Trans无 → Trans有 顺序取 MSE。"""
    order = [
        ("FNN", "无"),
        ("FNN", "有"),
        ("Transformer", "无"),
        ("Transformer", "有"),
    ]
    out = []
    for arch, phys in order:
        row = df[(df["贝叶斯方法"] == bayes) & (df["架构"] == arch) & (df["物理约束"] == phys)]
        if len(row) != 1:
            raise ValueError(f"未唯一匹配 MSE 行: {bayes}, {arch}, {phys}")
        out.append(float(row["MSE"].iloc[0]))
    return np.asarray(out, dtype=float)


def _pareto_annotate_label(row: pd.Series) -> str:
    key = (row["贝叶斯方法"], row["架构"], row["物理约束"])
    hints = {
        ("确定性", "FNN", "无"): "基线",
        ("确定性", "FNN", "有"): "+约束",
        ("确定性", "Transformer", "有"): "+Trans",
        ("α-VI", "Transformer", "有"): "+α-VI",
    }
    return hints.get(key, row["组合"][:16])


def plot_three_factor_dashboard(
    df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    fp_dejavu: FontProperties,
    fp_mixed: FontProperties,
    out_path: Path,
) -> None:
    """2×2：分组柱、时间–MSE 帕累托散点、热力图、性价比条形。"""
    colors = {"确定性": "#3498db", "α-VI": "#e74c3c"}
    marker_style: dict[str, str] = {"FNN": "o", "Transformer": "s"}

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "三因素实验分析",
        fontsize=18,
        fontweight="bold",
        y=0.98,
        fontproperties=fp_mixed,
    )

    # (a) 分组柱状图
    ax = axes[0, 0]
    x = np.arange(4)
    width = 0.35
    configs = ["FNN\n无约束", "FNN\n有约束", "Trans\n无约束", "Trans\n有约束"]
    det_mse = _mse_four_bar_profile(df, "确定性")
    alpha_mse = _mse_four_bar_profile(df, "α-VI")
    bars1 = ax.bar(
        x - width / 2,
        det_mse,
        width,
        label="确定性",
        color=colors["确定性"],
        alpha=0.8,
        edgecolor="black",
    )
    bars2 = ax.bar(
        x + width / 2,
        alpha_mse,
        width,
        label="α-VI",
        color=colors["α-VI"],
        alpha=0.8,
        edgecolor="black",
    )
    ax.set_ylabel("MSE (对数尺度)", fontsize=12, fontweight="bold", fontproperties=fp_mixed)
    ax.set_title(
        "(a) 精度对比：贝叶斯方法 × 架构 × 物理约束",
        fontsize=13,
        fontweight="bold",
        fontproperties=fp_mixed,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=11, fontproperties=fp_mixed)
    ax.legend(fontsize=11, prop=fp_mixed)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    for lbl in ax.get_yticklabels():
        lbl.set_fontproperties(fp_dejavu)
    for bar in bars1:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height * 1.08,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            fontproperties=fp_dejavu,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height * 1.08,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            fontproperties=fp_dejavu,
        )

    # (b) 时间–精度帕累托
    ax = axes[0, 1]
    for method in ["确定性", "α-VI"]:
        subset = df[df["贝叶斯方法"] == method]
        for arch in ["FNN", "Transformer"]:
            arch_data = subset[subset["架构"] == arch]
            marker = marker_style[arch]
            ax.scatter(
                arch_data["Time(s)"],
                arch_data["MSE"],
                s=np.clip(arch_data["Params"].to_numpy(dtype=float) / 100.0, 20.0, 800.0),
                c=colors[method],
                marker=marker,
                alpha=0.7,
                edgecolors="black",
                linewidths=1.5,
                label=f"{method}+{arch}",
            )
    p_sorted = pareto_df.sort_values("Time(s)")
    pareto_times = p_sorted["Time(s)"].to_numpy(dtype=float)
    pareto_mses = p_sorted["MSE"].to_numpy(dtype=float)
    ax.plot(
        pareto_times,
        pareto_mses,
        "k--",
        linewidth=2,
        alpha=0.6,
        label="帕累托前沿",
    )
    ax.set_xlabel("计算时间 (s)", fontsize=12, fontweight="bold", fontproperties=fp_mixed)
    ax.set_ylabel("MSE", fontsize=12, fontweight="bold", fontproperties=fp_dejavu)
    ax.set_title(
        "(b) 帕累托前沿分析\n(气泡大小=参数量)",
        fontsize=13,
        fontweight="bold",
        fontproperties=fp_mixed,
    )
    ax.set_yscale("log")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(fontsize=9, loc="upper right", prop=fp_mixed)
    for lbl in ax.get_xticklabels():
        lbl.set_fontproperties(fp_dejavu)
    for lbl in ax.get_yticklabels():
        lbl.set_fontproperties(fp_dejavu)
    for _, prow in p_sorted.iterrows():
        t, m = float(prow["Time(s)"]), float(prow["MSE"])
        ax.annotate(
            _pareto_annotate_label(prow),
            (t, m),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            fontproperties=fp_mixed,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )

    # (c) 热力图
    ax = axes[1, 0]
    pivot_mse = df.pivot_table(
        values="MSE",
        index=["架构", "贝叶斯方法"],
        columns="物理约束",
        aggfunc="mean",
    )
    arch_order = ["FNN", "Transformer"]
    bayes_order = ["确定性", "α-VI"]
    idx = pd.MultiIndex.from_tuples([(a, b) for a in arch_order for b in bayes_order])
    pivot_mse = pivot_mse.reindex(idx)
    col_order = [c for c in ("无", "有") if c in pivot_mse.columns]
    pivot_mse = pivot_mse[col_order]
    im = ax.imshow(pivot_mse.values, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(pivot_mse.columns)))
    ax.set_yticks(range(len(pivot_mse.index)))
    ax.set_xticklabels(pivot_mse.columns, fontsize=11, fontproperties=fp_mixed)
    ax.set_yticklabels([f"{idx[0]}\n{idx[1]}" for idx in pivot_mse.index], fontsize=10, fontproperties=fp_mixed)
    ax.set_title(
        "(c) MSE热力图\n(颜色越绿越好)",
        fontsize=13,
        fontweight="bold",
        fontproperties=fp_mixed,
    )
    for i in range(len(pivot_mse.index)):
        for j in range(len(pivot_mse.columns)):
            val = float(pivot_mse.values[i, j])
            ax.text(
                j,
                i,
                f"{val:.6f}",
                ha="center",
                va="center",
                color="white" if val < 0.0005 else "black",
                fontweight="bold",
                fontsize=9,
                fontproperties=fp_dejavu,
            )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("MSE", fontproperties=fp_dejavu)

    # (d) 性价比
    ax = axes[1, 1]
    df_sorted = df.sort_values("Cost_Efficiency", ascending=True)
    y_pos = np.arange(len(df_sorted))
    colors_bar = [colors[m] for m in df_sorted["贝叶斯方法"]]
    bars = ax.barh(y_pos, df_sorted["Cost_Efficiency"], color=colors_bar, alpha=0.7, edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [
            f"{row['贝叶斯方法']}+{row['架构']}+{row['物理约束']}"
            for _, row in df_sorted.iterrows()
        ],
        fontsize=10,
        fontproperties=fp_mixed,
    )
    # 勿在含中文的标签里使用 $...$，否则会整段走 mathtext，无法用 CJK 回退字体
    ax.set_xlabel(
        "性价比：-ln(MSE) / t（单位 1/s）",
        fontsize=12,
        fontweight="bold",
        fontproperties=fp_mixed,
    )
    ax.set_title(
        "(d) 性价比排序\n（-ln(MSE)/时间，越高越好）",
        fontsize=13,
        fontweight="bold",
        fontproperties=fp_mixed,
    )
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    for lbl in ax.get_xticklabels():
        lbl.set_fontproperties(fp_dejavu)
    xmax = float(df_sorted["Cost_Efficiency"].max())
    dx = max(xmax * 0.02, 1e-5)
    for i, (bar, val) in enumerate(zip(bars, df_sorted["Cost_Efficiency"])):
        ax.text(
            float(val) + dx,
            i,
            f"{float(val):.4f}",
            va="center",
            fontsize=10,
            fontweight="bold",
            fontproperties=fp_dejavu,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="2³ 析因：主效应 / 交互 / 性价比（数据来自 CSV）")
    parser.add_argument(
        "--csv",
        type=Path,
        default=_DEFAULT_CSV,
        help=f"因子实验表路径（默认: {_DEFAULT_CSV}）",
    )
    parser.add_argument(
        "--out-fig",
        type=Path,
        default=_SCRIPT_DIR / "experiments" / "marginal_effects.png",
        help="边际效应图（1×2）输出路径",
    )
    parser.add_argument(
        "--out-dashboard",
        type=Path,
        default=_SCRIPT_DIR / "experiments" / "three_factor_dashboard.png",
        help="三因素 2×2 综合图输出路径",
    )
    args = parser.parse_args()
    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"找不到 CSV: {csv_path}")

    df = load_factors_df(csv_path)

    print("=" * 70)
    print(f"数据概览（{csv_path}）")
    print("=" * 70)
    print(df[["组合", "Params", "Time(s)", "MSE"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("主效应分析（基于MSE - 越小越好）")
    print("=" * 70)

    factors = ["贝叶斯方法", "架构", "物理约束"]
    for factor in factors:
        effect = df.groupby(factor)["MSE"].mean()
        improvement = (1 - effect.min() / effect.max()) * 100
        print(f"\n{factor}:")
        for idx, val in effect.items():
            print(f"  {idx}: {val:.6f}")
        print(f"  → 改进幅度: {improvement:.1f}%")

    print("\n" + "=" * 70)
    print("计算时间成本分析")
    print("=" * 70)
    time_ratio = df.groupby("贝叶斯方法")["Time(s)"].mean()
    print(f"确定性平均时间: {time_ratio['确定性']:.1f}s")
    print(f"α-VI平均时间: {time_ratio['α-VI']:.1f}s")
    print(f"时间倍数: {time_ratio['α-VI']/time_ratio['确定性']:.1f}x")

    print("\n交互效应分析")
    print("=" * 70)

    ba = df.groupby(["贝叶斯方法", "架构"])["MSE"].mean().unstack()
    print("\n贝叶斯方法 × 架构:")
    print(ba)
    det_trans_imp = (1 - ba.loc["确定性", "Transformer"] / ba.loc["确定性", "FNN"]) * 100
    alpha_trans_imp = (1 - ba.loc["α-VI", "Transformer"] / ba.loc["α-VI", "FNN"]) * 100
    print(f"Transformer提升(确定性): {det_trans_imp:.1f}%")
    print(f"Transformer提升(α-VI): {alpha_trans_imp:.1f}%")
    print(f"协同效应: {alpha_trans_imp - det_trans_imp:+.1f}%")

    ac = df.groupby(["架构", "物理约束"])["MSE"].mean().unstack()
    print("\n架构 × 物理约束:")
    print(ac)
    fnn_const_imp = (1 - ac.loc["FNN", "有"] / ac.loc["FNN", "无"]) * 100
    trans_const_imp = (1 - ac.loc["Transformer", "有"] / ac.loc["Transformer", "无"]) * 100
    print(f"约束提升(FNN): {fnn_const_imp:.1f}%")
    print(f"约束提升(Transformer): {trans_const_imp:.1f}%")
    print(f"协同效应: {trans_const_imp - fnn_const_imp:+.1f}%")

    df = df.copy()
    # 性价比：MSE 越小则 -ln(MSE) 越大；除以时间得到「单位时间的对数精度」
    df["Cost_Efficiency"] = -np.log(df["MSE"].to_numpy(dtype=float)) / df["Time(s)"].to_numpy(
        dtype=float
    )

    print("\n" + "=" * 70)
    print("性价比分析（-ln(MSE) / 时间(s)，越大越好）")
    print("=" * 70)
    efficiency_df = df[
        ["组合", "贝叶斯方法", "架构", "物理约束", "Time(s)", "MSE", "Cost_Efficiency"]
    ].sort_values("Cost_Efficiency", ascending=False)
    print(efficiency_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("帕累托最优分析（精度 vs 时间）")
    print("=" * 70)
    pareto = []
    for i, row in df.iterrows():
        is_dominated = False
        for j, other in df.iterrows():
            if i != j:
                if other["Time(s)"] <= row["Time(s)"] and other["MSE"] <= row["MSE"]:
                    if other["Time(s)"] < row["Time(s)"] or other["MSE"] < row["MSE"]:
                        is_dominated = True
                        break
        if not is_dominated:
            pareto.append(row)

    pareto_df = pd.DataFrame(pareto).sort_values("Time(s)")
    print("\n帕累托前沿（无法被其他配置同时超越时间和精度）：")
    print(pareto_df[["组合", "Time(s)", "MSE"]].to_string(index=False))

    fp_dejavu, fp_mixed = _plot_fonts()
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "边际效应分析：因素间的协同与拮抗作用",
        fontsize=16,
        fontweight="bold",
        fontproperties=fp_mixed,
    )

    ax = axes[0]
    for arch in ["FNN", "Transformer"]:
        subset = df[df["架构"] == arch].sort_values("物理约束")
        ax.plot(
            subset["物理约束"],
            subset["MSE"],
            marker="o",
            linewidth=3,
            markersize=10,
            label=f"{arch}",
            alpha=0.8,
        )

        no_const = subset[subset["物理约束"] == "无"]["MSE"].values[0]
        yes_const = subset[subset["物理约束"] == "有"]["MSE"].values[0]
        improvement = (1 - yes_const / no_const) * 100
        mid_x = 0.5
        mid_y = (no_const + yes_const) / 2
        ax.annotate(
            f"↓{improvement:.0f}%",
            xy=(mid_x, mid_y),
            fontsize=11,
            fontweight="bold",
            ha="center",
            color="darkgreen" if arch == "Transformer" else "darkblue",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            fontproperties=fp_dejavu,
        )

    ax.set_ylabel("MSE", fontsize=12, fontweight="bold", fontproperties=fp_dejavu)
    ax.set_title(
        "(a) 物理约束 × 架构\n(线条斜率越大，协同越强)",
        fontsize=13,
        fontweight="bold",
        fontproperties=fp_mixed,
    )
    ax.legend(fontsize=11, prop=fp_dejavu)
    for lbl in ax.get_xticklabels():
        lbl.set_fontproperties(fp_mixed)
    for lbl in ax.get_yticklabels():
        lbl.set_fontproperties(fp_dejavu)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    ax = axes[1]
    for method in ["确定性", "α-VI"]:
        subset = df[df["贝叶斯方法"] == method].sort_values("架构")
        ax.plot(
            subset["架构"],
            subset["MSE"],
            marker="s",
            linewidth=3,
            markersize=10,
            label=f"{method}",
            alpha=0.8,
        )

        fnn = subset[subset["架构"] == "FNN"]["MSE"].values[0]
        trans = subset[subset["架构"] == "Transformer"]["MSE"].values[0]
        improvement = (1 - trans / fnn) * 100
        mid_x = 0.5
        mid_y = (fnn + trans) / 2
        ax.annotate(
            f"↓{improvement:.0f}%",
            xy=(mid_x, mid_y),
            fontsize=11,
            fontweight="bold",
            ha="center",
            color="darkred" if method == "α-VI" else "darkblue",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            fontproperties=fp_dejavu,
        )

    ax.set_ylabel("MSE", fontsize=12, fontweight="bold", fontproperties=fp_dejavu)
    ax.set_title(
        "(b) 架构 × 贝叶斯方法\n(α-VI下Transformer收益更大)",
        fontsize=13,
        fontweight="bold",
        fontproperties=fp_mixed,
    )
    ax.legend(fontsize=11, prop=fp_mixed)
    for lbl in ax.get_xticklabels():
        lbl.set_fontproperties(fp_dejavu)
    for lbl in ax.get_yticklabels():
        lbl.set_fontproperties(fp_dejavu)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_fig, dpi=150, bbox_inches="tight")

    plot_three_factor_dashboard(df, pareto_df, fp_dejavu, fp_mixed, args.out_dashboard.resolve())
    print(f"\n三因素综合图已保存: {args.out_dashboard.resolve()}")

    plt.show()
    plt.close("all")

    best_eff = efficiency_df.iloc[0]
    synergy_phys_arch = trans_const_imp - fnn_const_imp
    synergy_bayes_arch = alpha_trans_imp - det_trans_imp

    print("\n" + "=" * 70)
    print("关键发现总结（由当前 CSV 计算）")
    print("=" * 70)
    print(
        f"1. 物理约束：Transformer 上 {trans_const_imp:.1f}% MSE 改善，"
        f"FNN 上 {fnn_const_imp:.1f}%（协同增益 {synergy_phys_arch:+.1f}%）"
    )
    print(
        f"\n2. 架构 × 贝叶斯：确定性下 Transformer 相对 FNN {det_trans_imp:.1f}%，"
        f"α-VI 下 {alpha_trans_imp:.1f}%（协同增益 {synergy_bayes_arch:+.1f}%）"
    )
    print(
        f"\n3. 性价比（-ln(MSE)/时间）最高：{best_eff['组合']} "
        f"({best_eff['Cost_Efficiency']:.6f}，时间 {best_eff['Time(s)']:.1f}s）"
    )
    print("\n4. 帕累托前沿配置：")
    for _, prow in pareto_df.iterrows():
        print(f"   - {prow['组合']} ({prow['Time(s)']:.1f}s, MSE={prow['MSE']:.6f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
