"""汇总各阶段实验结果到 docs/tables，供论文引用。"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "docs" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def main():
    all_rows = []

    # Stage 3
    for p in (ROOT / "experiments" / "compare_4_combos").glob("stage3_*.csv"):
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                all_rows.append({"stage": "3", "case": "antiderivative", "bayes": r["bayes_method"], "branch": r["branch_type"], **r})

    # Stage 5
    for p in (ROOT / "experiments" / "ablation_antiderivative").glob("stage5_*.csv"):
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                all_rows.append({"stage": "5", "tag": r["tag"], **r})

    # Stage 6
    for p in (ROOT / "experiments" / "stage6").glob("stage6_summary_*.csv"):
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                all_rows.append({"stage": "6", **r})

    # Stage 7
    for p in (ROOT / "experiments" / "stage7").glob("stage7_summary_*.csv"):
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                all_rows.append({"stage": "7", **r})

    # Stage 8
    for p in (ROOT / "experiments" / "stage8").glob("stage8_summary_*.csv"):
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                all_rows.append({"stage": "8", **r})

    if not all_rows:
        print("[WARN] No result CSVs found. Run experiments first.")
        return

    out = TABLES / "paper_all_stages.csv"
    keys = list(all_rows[0].keys())
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"[SAVED] {out} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
