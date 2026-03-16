"""运行标准 epoch 实验：替换 ULTRA 快速验证，产出论文正式结果。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], desc: str, timeout_min: int = 60, env_override: dict | None = None) -> bool:
    print(f"\n{'='*60}\n{desc}\n{'='*60}")
    env = {**os.environ, "STAGE7_ULTRA": "0", "STAGE8_ULTRA": "0"}
    if env_override:
        env.update(env_override)
    ret = subprocess.run(cmd, cwd=ROOT, env=env, timeout=timeout_min * 60)
    return ret.returncode == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=str, default="all",
        help="3|4|5|6|7|8|all")
    p.add_argument("--epochs-stage6", type=int, default=15)
    p.add_argument("--epochs-single", type=int, default=30)
    p.add_argument("--checkpoint-every", type=int, default=5, help="每 N epoch 保存 checkpoint，0=禁用")
    p.add_argument("--seed", type=int, default=42, help="随机种子，保证可复现")
    p.add_argument("--resume", type=str, default=None, help="续训 checkpoint 路径（仅单 stage 时有效）")
    p.add_argument("--ultra", action="store_true", help="Stage 7/8 用 ULTRA（小数据，约 40min）")
    p.add_argument("--fast", action="store_true", help="Stage 7/8 用 fast（nx=30,nt=31）")
    p.add_argument("--faster", action="store_true", help="Stage 7/8 用 faster（nx=15,nt=16，约 30min/5epoch）")
    args = p.parse_args()

    stages = args.stage.split(",") if "," in args.stage else [args.stage]
    if "all" in stages:
        stages = ["3", "4", "5", "6", "7", "8"]

    ckpt = ["--checkpoint-every", str(args.checkpoint_every)] if args.checkpoint_every > 0 else []
    seed_args = ["--seed", str(args.seed)]
    resume_args = ["--resume", args.resume] if args.resume else []

    s78_mode = ["--faster"] if args.faster else (["--fast"] if args.fast else [])
    s78_env = {"STAGE7_ULTRA": "0", "STAGE8_ULTRA": "0"} if s78_mode else {"STAGE7_ULTRA": "1", "STAGE8_ULTRA": "1"}

    if "3" in stages:
        run([sys.executable, "scripts/compare_4_combos.py", "--epochs", str(args.epochs_single), *seed_args, *ckpt, *resume_args],
            "Stage 3: branch × bayes 四组合", timeout_min=45)

    if "4" in stages:
        # Stage 4 含于 Stage 5：compare_ablation 的 pi_constraint 维度即 stage 4
        pass

    if "5" in stages:
        run([sys.executable, "scripts/compare_ablation.py", "--config", "configs/paper_antiderivative.yaml",
             "--case", "antiderivative", "--epochs", str(args.epochs_single), *seed_args, *ckpt, *resume_args],
            "Stage 5: 单输出综合消融", timeout_min=90)

    if "6" in stages:
        run([sys.executable, "scripts/run_stage6_experiments.py", "--route", "all", "--branch", "both",
             "--epochs", str(args.epochs_stage6), *seed_args, *ckpt, *resume_args],
            "Stage 6: 多输出 NS (标准 epoch)", timeout_min=120)

    if "7" in stages:
        run([sys.executable, "scripts/run_stage7_experiments.py", "--epochs", str(args.epochs_single),
             *s78_mode, *seed_args, *ckpt, *resume_args],
            "Stage 7: PI 三模式", timeout_min=90, env_override=s78_env)

    if "8" in stages:
        run([sys.executable, "scripts/run_stage8_experiments.py", "--epochs", str(args.epochs_single),
             *s78_mode, *seed_args, *ckpt, *resume_args],
            "Stage 8: Trunk 四模式", timeout_min=120, env_override=s78_env)

    print("\n[DONE] 标准实验完成。运行 uv run python scripts/plot_paper_figures.py 更新图表。")


if __name__ == "__main__":
    main()
