# Stage 7: PI Extension Commands

## Physics modes

- `standard_pi`: data + PDE residual + soft BC/IC loss
- `hard_bc_pi`: output transform enforces Dirichlet BC/IC; no BC/IC loss
- `stabilized_pi`: subdomain-averaged PDE residual (stabilized PI)

## Configs

- `configs/diffusion_reaction_standard_pi.yaml`
- `configs/diffusion_reaction_hard_bc.yaml`
- `configs/diffusion_reaction_stabilized_pi.yaml`

## Run single experiment

```bash
uv run python main.py --config configs/diffusion_reaction_standard_pi.yaml
uv run python main.py --config configs/diffusion_reaction_hard_bc.yaml
uv run python main.py --config configs/diffusion_reaction_stabilized_pi.yaml
```

## Run all three (stage7 experiment script)

```bash
uv run python scripts/run_stage7_experiments.py
```

Results CSV: `experiments/stage7/stage7_summary_epochs30.csv`
