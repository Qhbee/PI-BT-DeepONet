"""阶段6 正式实验脚本：路线A/B × hard-truncation vs multi-CLS，输出参数量/时间/loss/测试误差表格。"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# 直接 `python scripts/run_stage6_experiments.py` 时使用的默认；命令行仍可覆盖。
DEFAULT_EPOCHS = 15
DEFAULT_CONFIG_DIR = str(_REPO_ROOT / "configs")
DEFAULT_OUT_DIR = str(_REPO_ROOT / "experiments" / "stage6")
DEFAULT_ROUTE = "all"  # A | B | all
DEFAULT_BRANCH = "both"  # hard | cls | both
DEFAULT_FAST = True
DEFAULT_ULTRA = False
DEFAULT_CHECKPOINT_EVERY = 0
DEFAULT_RESUME: str | None = None
DEFAULT_SEED: int | None = None

from src.data import get_generator
from src.data.generators import (  # noqa: F401
    generate_ns_beltrami_ic2field_data,
    generate_ns_beltrami_parametric_data,
    generate_ns_kovasznay_bc2field_data,
    generate_ns_kovasznay_parametric_data,
)
from src.models import (
    MultiOutputDeepONet,
    TransformerMultiCLSBranch,
    TransformerMultiOutputBranch,
    FNNTrunk,
)
from src.training import train_operator


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _write_csv(out_dir: Path, results: list, epochs: int) -> None:
    """无论何时退出都写入 CSV（含部分结果）。"""
    if not results:
        return
    csv_path = out_dir / f"stage6_summary_epochs{epochs}.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("case,branch,params,time_s,loss,rel_l2,test_mse\n")
        for r in results:
            f.write(f"{r['case']},{r['branch']},{r['params']},{r['time_s']:.2f},{r['loss']:.6f},{r['rel_l2']:.6f},{r['test_mse']:.6f}\n")
    print(f"\n[CSV] {len(results)} 条结果已保存到 {csv_path}")


def build_model(cfg: dict, coord_dim: int, n_outputs: int, input_channels: int, branch_type: str):
    model_cfg = dict(cfg.get("model", {}))
    train_cfg = cfg.get("training", {})
    num_sensors = int(model_cfg.get("num_sensors", 64))
    p_group = int(model_cfg.get("p_group", 32))
    branch_hidden = model_cfg.get("branch_hidden", [64, 64])
    trunk_hidden = model_cfg.get("trunk_hidden", [64, 64])

    if branch_type == "transformer_multicls":
        branch = TransformerMultiCLSBranch(
            num_sensors=num_sensors,
            n_outputs=n_outputs,
            p_group=p_group,
            d_model=model_cfg.get("transformer_d_model", 64),
            nhead=model_cfg.get("transformer_nhead", 4),
            num_layers=model_cfg.get("transformer_num_layers", 2),
            dropout=model_cfg.get("transformer_dropout", 0.1),
            input_channels=input_channels,
        )
    else:
        branch = TransformerMultiOutputBranch(
            num_sensors=num_sensors,
            n_outputs=n_outputs,
            p_group=p_group,
            d_model=model_cfg.get("transformer_d_model", 64),
            nhead=model_cfg.get("transformer_nhead", 4),
            num_layers=model_cfg.get("transformer_num_layers", 2),
            dropout=model_cfg.get("transformer_dropout", 0.1),
            input_channels=input_channels,
        )
    trunk = FNNTrunk(coord_dim, trunk_hidden, n_outputs * p_group)
    model = MultiOutputDeepONet(
        branch=branch,
        trunk=trunk,
        n_outputs=n_outputs,
        p_group=p_group,
        bias=True,
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--config-dir", type=str, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--route", type=str, choices=["A", "B", "all"], default=DEFAULT_ROUTE)
    parser.add_argument("--branch", type=str, choices=["hard", "cls", "both"], default=DEFAULT_BRANCH)
    parser.add_argument(
        "--fast",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FAST,
        help="快速模式：减少 n_train/nx/ny/n_collocation，加速约 10x",
    )
    parser.add_argument(
        "--ultra",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ULTRA,
        help="极速模式：每实验约 30s，n_train=20, nx=ny=16, n_collocation=16",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY, help="每 N epoch 保存 checkpoint，0=禁用"
    )
    parser.add_argument("--resume", type=str, default=DEFAULT_RESUME, help="续训 checkpoint 路径")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子，覆盖 data_cfg")
    args = parser.parse_args()

    if args.ultra:
        args.fast = True  # ultra 继承 fast 的数据缩减

    # 实验矩阵：路线A → 路线B，每种分支 hard/CLS
    route_a_cases = ["ns_kovasznay_parametric", "ns_beltrami_parametric"]
    route_b_cases = ["ns_kovasznay_bc2field", "ns_beltrami_ic2field"]

    if args.route == "A":
        cases = route_a_cases
    elif args.route == "B":
        cases = route_b_cases
    else:
        cases = route_a_cases + route_b_cases

    branch_types = []
    if args.branch in ("hard", "both"):
        branch_types.append(("transformer_multi_output", "hard"))
    if args.branch in ("cls", "both"):
        branch_types.append(("transformer_multicls", "cls"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    if args.ultra:
        print("Ultra mode: Kovasznay 16×16; Beltrami nt=2,nx=ny=nz=6,n_coll=8 (~30s/实验)")
    elif args.fast:
        print("Fast mode: n_train=50, nx=ny=32, n_collocation=64, batch_size=256")

    results = []

    def _flush_csv():
        _write_csv(out_dir, results, args.epochs)

    try:
        for case in cases:
            cfg_path = Path(args.config_dir) / f"{case}.yaml"
            if not cfg_path.exists():
                print(f"Skip {case}: config not found")
                continue

            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}

            model_cfg = cfg.get("model", {})
            train_cfg = dict(cfg.get("training", {}))
            physics_cfg = dict(cfg.get("physics", {}))
            data_cfg = dict(cfg.get("data", {}))

            if args.ultra:
                data_cfg.update(n_train=20, n_test=10, nx=16, ny=16)
                if case.startswith("ns_beltrami"):
                    # Beltrami 4D(t,x,y,z)：点数 nt×nx×ny×nz，VP 残差 4 变量×4 维梯度，计算量远大于 2D Kovasznay
                    data_cfg.update(nt=2, nx=6, ny=6, nz=6)  # 432 点/样本，约 50x 缩减
                    physics_cfg["n_collocation"] = 8
                    train_cfg["batch_size"] = 128
                else:
                    physics_cfg["n_collocation"] = 16
                    train_cfg["batch_size"] = 512
            elif args.fast:
                data_cfg.update(n_train=50, n_test=20, nx=32, ny=32)
                if case.startswith("ns_beltrami"):
                    data_cfg.update(nt=3, nx=10, ny=10, nz=10)  # 3000 点/样本
                    physics_cfg["n_collocation"] = 32
                    train_cfg["batch_size"] = 64
                else:
                    physics_cfg["n_collocation"] = 64
                    train_cfg["batch_size"] = 256

            if args.seed is not None:
                data_cfg["seed"] = args.seed
            generator = get_generator(case)
            if "n_sensors" in inspect.signature(generator).parameters:
                data_cfg.setdefault("n_sensors", int(model_cfg.get("num_sensors", 32 if args.ultra else 64)))
            data = generator(**data_cfg)

            y_train = data["y_train"]
            s_train = data["s_train"]
            u_train = data["u_train"]
            coord_dim = int(y_train.shape[-1]) if y_train.ndim == 3 else 1
            n_outputs = int(s_train.shape[-1]) if s_train.ndim == 3 else 1
            if u_train.ndim == 3:
                input_channels = int(u_train.shape[-1])
            elif case.startswith("ns_") and u_train.ndim == 2 and u_train.shape[1] <= 4:
                input_channels = int(u_train.shape[1])
            else:
                input_channels = int(model_cfg.get("input_channels", 1))

            for branch_type, branch_name in branch_types:
                tag = f"{case}_{branch_name}"
                model = build_model(cfg, coord_dim, n_outputs, input_channels, branch_type)
                n_params = count_params(model)

                t0 = time.perf_counter()
                log_dir = out_dir / tag
                resume_path = args.resume if (args.resume and tag in args.resume) else None
                _, metrics = train_operator(
                    model=model,
                    data=data,
                    case=case,
                    lr=train_cfg.get("lr", 0.001),
                    epochs=args.epochs,
                    batch_size=train_cfg.get("batch_size", 64),
                    log_dir=str(log_dir),
                    device=device,
                    bayes_method="deterministic",
                    alpha=train_cfg.get("alpha", 1.0),
                    mc_samples=train_cfg.get("mc_samples", 3),
                    eval_mc_samples=train_cfg.get("eval_mc_samples", 20),
                    kl_weight=None,
                    pi_constraint=physics_cfg.get("pi_constraint", "none"),
                    pi_weight=physics_cfg.get("pi_weight", 0.0),
                    bc_weight=physics_cfg.get("bc_weight", 0.0),
                    ic_weight=physics_cfg.get("ic_weight", 0.0),
                    n_collocation=physics_cfg.get("n_collocation", 256),
                    diffusion_D=physics_cfg.get("diffusion_D", 0.01),
                    reaction_k=physics_cfg.get("reaction_k", 0.1),
                    burgers_nu=physics_cfg.get("burgers_nu", 0.01 / torch.pi),
                    ns_nu=physics_cfg.get("ns_nu", 1.0 / 40.0),
                    ns_beltrami_nu=physics_cfg.get("ns_beltrami_nu", 1.0),
                    pressure_gauge_weight=physics_cfg.get("pressure_gauge_weight", 0.0),
                    checkpoint_every=args.checkpoint_every,
                    resume_from=resume_path,
                    seed=data_cfg.get("seed"),
                )
                elapsed = time.perf_counter() - t0

                hist = metrics.get("history", [])
                if hist:
                    hist_path = log_dir / "training_history.json"
                    hist_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(hist_path, "w", encoding="utf-8") as f:
                        json.dump(hist, f, indent=2)
                    print(f"    [History] {len(hist)} 条 -> {hist_path}")

                results.append({
                    "case": case,
                    "branch": branch_name,
                    "params": n_params,
                    "time_s": elapsed,
                    "loss": metrics["loss"],
                    "rel_l2": metrics["rel_l2"],
                    "test_mse": metrics["test_mse"],
                })
                _write_csv(out_dir, results, args.epochs)  # 每完成一个实验就写 CSV
                print(f"  Done: {tag} ({elapsed:.1f}s)")
    except (KeyboardInterrupt, Exception) as e:
        print(f"\n[中断] {e}")
    finally:
        _flush_csv()

    # 表格输出
    if results:
        print("\n" + "=" * 120)
        print(f"阶段6 实验结果 (epochs={args.epochs}, 共 {len(results)} 条)")
        print("=" * 120)
        print(f"{'Case':<30} {'Branch':<8} {'Params':>12} {'Time(s)':>10} {'Loss':>12} {'RelL2':>12} {'TestMSE':>14}")
        print("-" * 120)
        for r in results:
            print(
                f"{r['case']:<30} {r['branch']:<8} {r['params']:>12,} {r['time_s']:>10.1f} "
                f"{r['loss']:>12.6f} {r['rel_l2']:>12.6f} {r['test_mse']:>14.6f}"
            )
        print("=" * 120)


if __name__ == "__main__":
    main()
