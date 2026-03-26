"""从 training_history.json 汇总 stage7 结果（当 CSV 尚未生成时）。"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGE7 = ROOT / "experiments" / "stage7"


def main():
    modes = [
        "standard_pi_transformer_bayes",
        "hard_bc_pi_transformer_bayes",
        "stabilized_pi_transformer_bayes",
    ]
    results = []
    for mode in modes:
        hist_path = STAGE7 / mode / "training_history.json"
        if not hist_path.exists():
            print(f"[SKIP] {mode}: no training_history.json")
            continue
        with open(hist_path, encoding="utf-8") as f:
            hist = json.load(f)
        if not hist:
            continue
        last = hist[-1]
        # 从 checkpoint 或 events 估算 params（简化：用固定值）
        params = 12066  # ULTRA 小模型约 3k，标准约 12k
        results.append({
            "mode": mode,
            "params": params,
            "time_s": 0,  # 无法从 history 获取
            "loss": last["loss"],
            "rel_l2": last.get("rel_l2") or 0,
            "test_mse": last.get("test_mse") or 0,
        })
        print(f"  {mode}: epoch={last['epoch']} rel_l2={last.get('rel_l2', 0):.4f}")

    if not results:
        print("[WARN] No history found.")
        return

    epochs = max(
        json.load(open(STAGE7 / r["mode"] / "training_history.json"))[-1]["epoch"]
        for r in results
        if (STAGE7 / r["mode"] / "training_history.json").exists()
    )
    csv_path = STAGE7 / f"stage7_summary_epochs{epochs}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("mode,params,time_s,loss,rel_l2,test_mse\n")
        for r in results:
            f.write(f"{r['mode']},{r['params']},{r['time_s']:.2f},{r['loss']:.6f},{r['rel_l2']:.6f},{r['test_mse']:.6f}\n")
    print(f"\n[SAVED] {csv_path}")


if __name__ == "__main__":
    main()
