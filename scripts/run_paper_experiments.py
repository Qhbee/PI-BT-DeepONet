"""Run paper experiments via main.py and collect results."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure we collect history - trainer already returns it
# We run main.py and parse tensorboard, or we run a custom loop
# Simpler: run experiments and copy existing CSVs + run stage7/8

def main():
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    tables_dir = Path("docs/tables")
    figures_dir = Path("docs/figures")
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("configs/paper_antiderivative.yaml", "antiderivative_fnn"),
        ("configs/paper_diffusion_reaction.yaml", "diffusion_reaction_transformer_vi"),
    ]
    results = []

    for cfg_path, run_name in configs:
        if not Path(cfg_path).exists():
            print(f"[SKIP] {cfg_path}")
            continue
        print(f"\n{'='*60}\nRunning {run_name}\n{'='*60}")
        ret = subprocess.run(
            [sys.executable, "main.py", "--config", cfg_path],
            cwd=root,
            capture_output=False,
        )
        if ret.returncode != 0:
            print(f"[WARN] {run_name} exited with {ret.returncode}")

    # Run stage 7 (PI extension)
    print("\n" + "=" * 60 + "\nStage 7: PI extension\n" + "=" * 60)
    os.environ["STAGE7_ULTRA"] = "0"
    subprocess.run([sys.executable, "scripts/run_stage7_experiments.py"], cwd=root)

    # Run stage 8 (Trunk extension)
    print("\n" + "=" * 60 + "\nStage 8: Trunk extension\n" + "=" * 60)
    os.environ["STAGE8_ULTRA"] = "0"
    subprocess.run([sys.executable, "scripts/run_stage8_experiments.py"], cwd=root)

    # Collect stage6 if exists
    s6 = Path("experiments/stage6/stage6_summary_epochs5.csv")
    if s6.exists():
        with open(s6, encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                results.append({"source": "stage6", **row})

    s7 = Path("experiments/stage7/stage7_summary_epochs2.csv")
    if not s7.exists():
        s7 = Path("experiments/stage7/stage7_summary_epochs15.csv")
    if s7.exists():
        with open(s7, encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                results.append({"source": "stage7", **row})

    s8 = Path("experiments/stage8/stage8_summary_epochs5.csv")
    if not s8.exists():
        s8 = Path("experiments/stage8/stage8_summary_epochs15.csv")
    if s8.exists():
        with open(s8, encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                results.append({"source": "stage8", **row})

    # Save combined table
    out_csv = tables_dir / "paper_all_results.csv"
    if results:
        keys = list(results[0].keys())
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(results)
        print(f"\n[SAVED] {out_csv} ({len(results)} rows)")
    return results

if __name__ == "__main__":
    main()
