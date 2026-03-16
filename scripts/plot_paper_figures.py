"""Generate paper figures from tensorboard logs, training_history.json and CSV results."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "docs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# 时间戳后缀，方便对比不同运行
TS = datetime.now().strftime("%Y%m%d_%H%M%S")


def load_tb_scalars(log_dir: Path, tag: str) -> list[tuple[int, float]]:
    """Load scalar values from tensorboard events."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
        ea = event_accumulator.EventAccumulator(str(log_dir))
        ea.Reload()
        if tag not in ea.Tags().get("scalars", []):
            return []
        return [(e.step, e.value) for e in ea.Scalars(tag)]
    except Exception:
        return []


def load_history_json(path: Path) -> list[dict] | None:
    """Load training history from JSON (epoch, loss, rel_l2, test_mse)."""
    p = path / "training_history.json" if path.is_dir() else path
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _get_curves(hist: list[dict], key: str) -> list[tuple[int, float]]:
    """Extract (epoch, value) from history for given key."""
    out = []
    for h in hist:
        v = h.get(key)
        if v is not None:
            out.append((int(h.get("epoch", len(out) + 1)), float(v)))
    return out


def main():
    # 1. Training curves from antiderivative and diffusion_reaction
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    antiderivative_data = None
    for ax, (name, log_path) in zip(axes, [
        ("Antiderivative", ROOT / "experiments" / "antiderivative"),
        ("Diffusion-Reaction", ROOT / "experiments" / "diffusion_reaction"),
    ]):
        loss_data, rel_data = [], []
        hist = load_history_json(log_path) if name == "Antiderivative" else None
        if name == "Antiderivative" and hist:
            antiderivative_data = hist
            loss_data = _get_curves(hist, "loss")
            rel_data = _get_curves(hist, "rel_l2")
        if not loss_data and log_path.exists():
            loss_data = load_tb_scalars(log_path, "loss/train")
            rel_data = load_tb_scalars(log_path, "metric/rel_l2")
        if name == "Antiderivative" and not loss_data:
            # fallback: ablation antiderivative_deterministic_fnn_none
            for sub in ["antiderivative_deterministic_fnn_none", "deterministic_fnn"]:
                h = load_history_json(ROOT / "experiments" / "ablation_antiderivative" / sub) or load_history_json(ROOT / "experiments" / "compare_4_combos" / sub)
                if h:
                    antiderivative_data = h
                    loss_data = _get_curves(h, "loss")
                    rel_data = _get_curves(h, "rel_l2")
                    break
        if name == "Antiderivative" and loss_data and not antiderivative_data:
            # tb data: build pseudo history for log plot
            antiderivative_data = [{"epoch": s, "loss": v, "rel_l2": next((v2 for e2, v2 in rel_data if e2 == s), None)} for s, v in loss_data]
        if loss_data:
            steps, vals = zip(*loss_data)
            ax.plot(steps, vals, "b-", alpha=0.8, label="Loss")
        if rel_data:
            steps, vals = zip(*rel_data)
            ax2 = ax.twinx()
            ax2.plot(steps, vals, "g--", alpha=0.7, label="RelL2")
            ax2.set_ylabel("RelL2", color="green")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    fig.tight_layout()
    for suf in ["", f"_{TS}"]:
        fig.savefig(FIGS / f"training_curves{suf}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {FIGS / 'training_curves.png'} + training_curves_{TS}.png")

    # 1b. Antiderivative 对数 y 轴（左图不清楚时便于观察）
    if antiderivative_data:
        loss_curve = _get_curves(antiderivative_data, "loss")
        rel_curve = _get_curves(antiderivative_data, "rel_l2")
        fig, ax = plt.subplots(figsize=(5, 4))
        steps = None
        if loss_curve:
            steps, vals = zip(*loss_curve)
            ax.semilogy(steps, [max(v, 1e-8) for v in vals], "b-", lw=0.5, alpha=0.9, label="Loss")
        if rel_curve:
            ax2 = ax.twinx()
            r_steps, r_vals = zip(*rel_curve)
            ax2.plot(r_steps, r_vals, "g--", lw=0.5, alpha=0.9, label="RelL2")
            ax2.set_ylabel("RelL2", color="green")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (log scale)")
        ax.set_title("Antiderivative (log scale)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        fig.tight_layout()
        for suf in ["", f"_{TS}"]:
            fig.savefig(FIGS / f"training_curves_antiderivative_log{suf}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SAVED] training_curves_antiderivative_log.png + _{TS}")

    # 2. Stage 6 bar chart
    s6 = ROOT / "experiments" / "stage6" / "stage6_summary_epochs5.csv"
    if s6.exists():
        rows = list(csv.DictReader(open(s6, encoding="utf-8")))
        cases = list(dict.fromkeys(r["case"] for r in rows))
        hard_vals = [next(float(r["rel_l2"]) for r in rows if r["case"] == c and r["branch"] == "hard") for c in cases]
        cls_vals = [next(float(r["rel_l2"]) for r in rows if r["case"] == c and r["branch"] == "cls") for c in cases]
        case_names = [c.replace("ns_", "").replace("_parametric", " (A)").replace("_bc2field", " (B)").replace("_icbc2field", " (B)")[:18] for c in cases]
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(case_names))
        w = 0.35
        ax.bar(x - w/2, hard_vals, w, label="hard", color="steelblue")
        ax.bar(x + w/2, cls_vals, w, label="cls", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels(case_names, fontsize=9)
        ax.set_ylabel("RelL2")
        ax.set_title("Stage 6: Multi-output NS (5 epochs)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        for sfx in ["", f"_{TS}"]:
            fig.savefig(FIGS / f"stage6_rel_l2{sfx}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SAVED] stage6_rel_l2.png + _{TS}")

    # 3. Stage 7 bar chart (prefer epochs30, fallback epochs2)
    s7 = ROOT / "experiments" / "stage7" / "stage7_summary_epochs30.csv"
    if not s7.exists():
        s7 = ROOT / "experiments" / "stage7" / "stage7_summary_epochs2.csv"
    if s7.exists():
        modes, rel_l2s = [], []
        with open(s7, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                modes.append(row["mode"].replace("_transformer_bayes", ""))
                rel_l2s.append(float(row["rel_l2"]))
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(modes, rel_l2s, color=["#2ecc71", "#e74c3c", "#3498db"], alpha=0.8)
        ax.set_ylabel("RelL2")
        ax.set_title("Stage 7: PI extension (standard_pi / hard_bc_pi / s_pinn)")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        for suf in ["", f"_{TS}"]:
            fig.savefig(FIGS / f"stage7_rel_l2{suf}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SAVED] stage7_rel_l2.png + _{TS}")

    # 3b. Stage 7 训练曲线（便于分析高 RelL2 原因）
    stage7_dirs = [
        (ROOT / "experiments" / "stage7" / "standard_pi_transformer_bayes", "standard_pi"),
        (ROOT / "experiments" / "stage7" / "hard_bc_pi_transformer_bayes", "hard_bc_pi"),
        (ROOT / "experiments" / "stage7" / "s_pinn_transformer_bayes", "s_pinn"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for d, label in stage7_dirs:
        loss_d = load_tb_scalars(d, "loss/train")
        rel_d = load_tb_scalars(d, "metric/rel_l2")
        if loss_d:
            steps, vals = zip(*loss_d)
            axes[0].plot(steps, vals, "-", alpha=0.8, label=label)
        if rel_d:
            steps, vals = zip(*rel_d)
            axes[1].plot(steps, vals, "-", alpha=0.8, label=label)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Stage 7: Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RelL2")
    axes[1].set_title("Stage 7: RelL2")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    for suf in ["", f"_{TS}"]:
        fig.savefig(FIGS / f"stage7_training_curves{suf}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] stage7_training_curves.png + _{TS}")

    # 4. Stage 8 bar chart (if exists)
    for suf in ["epochs5", "epochs15"]:
        s8 = ROOT / "experiments" / "stage8" / f"stage8_summary_{suf}.csv"
        if s8.exists():
            modes, rel_l2s = [], []
            with open(s8, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    m = row.get("mode", "")
                    modes.append(m.replace("trunk_", "").replace("_transformer_bayes", "") if m else "?")
                    rel_l2s.append(float(row.get("rel_l2", 0)))
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(modes, rel_l2s, color="teal", alpha=0.8)
            ax.set_ylabel("RelL2")
            ax.set_title("Stage 8: Trunk extension (FNN / POD / Ex / ExV2)")
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            for sfx in ["", f"_{TS}"]:
                fig.savefig(FIGS / f"stage8_rel_l2{sfx}.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[SAVED] stage8_rel_l2.png + _{TS}")
            break

    # 4b. Stage 8 训练曲线（便于分析高 RelL2 原因）
    stage8_dirs = [
        (ROOT / "experiments" / "stage8" / "trunk_fnn_transformer_bayes", "fnn"),
        (ROOT / "experiments" / "stage8" / "trunk_pod_transformer_bayes", "pod"),
        (ROOT / "experiments" / "stage8" / "trunk_ex_transformer_bayes", "ex"),
        (ROOT / "experiments" / "stage8" / "trunk_ex_v2_transformer_bayes", "ex_v2"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for d, label in stage8_dirs:
        loss_d = load_tb_scalars(d, "loss/train")
        rel_d = load_tb_scalars(d, "metric/rel_l2")
        if loss_d:
            steps, vals = zip(*loss_d)
            axes[0].plot(steps, vals, "-", alpha=0.8, label=label)
        if rel_d:
            steps, vals = zip(*rel_d)
            axes[1].plot(steps, vals, "-", alpha=0.8, label=label)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Stage 8: Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RelL2")
    axes[1].set_title("Stage 8: RelL2")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    for suf in ["", f"_{TS}"]:
        fig.savefig(FIGS / f"stage8_training_curves{suf}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] stage8_training_curves.png + _{TS}")


if __name__ == "__main__":
    main()
