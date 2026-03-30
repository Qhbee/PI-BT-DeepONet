"""从各 run 的 training_history.json 取 rel_l2、test_mse 全程最小值，写回 stage6 汇总 CSV。

loss、params、time_s 及 case/branch 列保持不变；仅覆盖 rel_l2、test_mse 两列。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STAGE6 = _REPO_ROOT / "experiments" / "stage6"
DEFAULT_CSV = DEFAULT_STAGE6 / "stage6_summary_epochs20.csv"


def _run_folder(case: str, branch: str) -> str:
    return f"{case}_{branch}"


def _min_metrics(history_path: Path) -> tuple[float, float]:
    with history_path.open(encoding="utf-8") as f:
        hist = json.load(f)
    rel = [float(h["rel_l2"]) for h in hist]
    tm = [float(h["test_mse"]) for h in hist]
    return min(rel), min(tm)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="要更新的汇总 CSV 路径")
    p.add_argument("--stage6-dir", type=Path, default=DEFAULT_STAGE6, help="experiments/stage6 根目录")
    p.add_argument("--dry-run", action="store_true", help="只打印新值，不写回文件")
    args = p.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"文件不存在: {args.csv}")

    rows: list[dict[str, str]] = []
    with args.csv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise SystemExit("CSV 无表头")
        for row in reader:
            rows.append(dict(row))

    updated = 0
    for row in rows:
        case = row.get("case", "").strip()
        branch = row.get("branch", "").strip()
        if not case or not branch:
            continue
        folder = _run_folder(case, branch)
        hist_path = args.stage6_dir / folder / "training_history.json"
        if not hist_path.is_file():
            print(f"[warn] 缺失 {hist_path}，跳过该行 rel_l2/test_mse 更新")
            continue
        rel_min, mse_min = _min_metrics(hist_path)
        row["rel_l2"] = f"{rel_min:.6f}"
        row["test_mse"] = f"{mse_min:.6f}"
        updated += 1
        print(f"{folder}: rel_l2_min={rel_min:.6f}, test_mse_min={mse_min:.6f}")

    if args.dry_run:
        print(f"[dry-run] 共 {updated} 行将写回（未写入磁盘）")
        return

    with args.csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    print(f"[saved] {args.csv}（已更新 {updated} 行的 rel_l2 / test_mse 最小值）")


if __name__ == "__main__":
    main()
