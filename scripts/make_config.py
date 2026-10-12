"""Generate a run config by overriding fields of a base config.

Why this exists: every swept run needs its own yaml. Hand-editing per run
silently invites two failure modes, both relevant to this project:

  1. early stopping cutting the fixed budget short -- the base config has
     epochs=50, patience=15, and train_hmacd.py breaks at patience_counter
     >= patience, so any config that forgets to raise patience stops early
     and the fixed-budget protocol (proposal §6.1/§6.2) silently fails;
  2. configs that look different but overwrite the same checkpoint because
     experiment_name() collapses them (see src/utils/exp_name.py).

So this script (proposal 附录 C):
  - reads a base config (default configs/hmacd_resnet1d.yaml),
  - applies only the fields you pass,
  - forces patience >= epochs (refuses a smaller explicit value),
  - refuses to activate BOTH diversity families at once -- that hybrid is
    not a registered Gate A configuration (proposal 附录 G arms are:
    无 diversity / 平均余弦 / 去相关, exactly one family),
  - derives the output path from the experiment name unless --out is given,
  - prints the resulting experiment name so collisions are visible.

Generated files carry values only; comments live in the base config.

Run from the repo root. Usage:

    # Gate A arm 1, candidate lambda_d = 0.05, seed 42
    python scripts/make_config.py --seed 42 --lambda-diverse 0.05 --epochs 50

    # Gate A arm 2, candidate lambda_decor = 0.2 (diversity must be 0)
    python scripts/make_config.py --seed 42 --lambda-decor 0.2 --epochs 50
"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.exp_name import experiment_name  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="configs/hmacd_resnet1d.yaml",
                        help="Base config to override")
    parser.add_argument("--out", type=str, default=None,
                        help="Output path. Default: configs/<experiment_name>.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-concepts", type=int, default=None)
    parser.add_argument("--arm", type=str, default=None,
                        choices=["global", "scatter", "segment"],
                        help="Evidence-span arm (Step 2). global = all L "
                             "positions; scatter = arm_positions spread "
                             "positions shared by all concepts; segment = "
                             "segment_m contiguous groups, K/M concepts each.")
    parser.add_argument("--arm-positions", type=int, default=None,
                        help="Candidate count for --arm scatter (default 8).")
    parser.add_argument("--segment-m", type=int, default=None,
                        help="Number of contiguous segments for --arm segment "
                             "(default 4).")
    parser.add_argument("--lambda-diverse", type=float, default=None)
    parser.add_argument("--lambda-compact", type=float, default=None)
    parser.add_argument("--lambda-decor", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None,
                        help="Default: auto-set equal to epochs (fixed budget, "
                             "early stopping disabled). A smaller explicit value "
                             "is refused.")
    args = parser.parse_args()

    with open(args.base) as f:
        config = yaml.safe_load(f)

    def set_if(path, value):
        """Set config[path] only when value is not None."""
        if value is None:
            return
        d = config
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = value

    set_if(["training", "seed"], args.seed)
    set_if(["model", "arm"], args.arm)
    set_if(["model", "arm_positions"], args.arm_positions)
    set_if(["model", "segment_m"], args.segment_m)
    set_if(["training", "lambda_diverse"], args.lambda_diverse)
    set_if(["training", "lambda_compact"], args.lambda_compact)
    set_if(["training", "lambda_decor"], args.lambda_decor)
    set_if(["training", "epochs"], args.epochs)
    set_if(["model", "num_concepts"], args.num_concepts)

    train = config["training"]

    # ---- fixed budget: patience must be >= epochs (proposal §6.1/§6.2) ----
    if args.patience is not None and args.patience < int(train.get("epochs", 50)):
        parser.error(f"--patience {args.patience} < epochs {train['epochs']}: "
                     f"early stopping would cut the fixed budget short")
    train["patience"] = (args.patience if args.patience is not None
                         else int(train.get("epochs", 50)))

    # ---- one diversity family at a time (proposal 附录 G arms) ----
    # The decorrelation family REPLACES the mean-cosine one, so a positive
    # --lambda-decor with no explicit --lambda-diverse zeroes the latter
    # (the base config's lambda_diverse=0.05 must not sneak into arm 2).
    ld = float(train.get("lambda_diverse", 0.0))
    ldec = float(train.get("lambda_decor", 0.0))
    if ldec > 0 and args.lambda_diverse is None:
        train["lambda_diverse"] = 0.0
        ld = 0.0
        print("note: lambda_diverse auto-set to 0 (the decorrelation family "
              "replaces mean-cosine; pass --lambda-diverse explicitly if you "
              "really want both, which will then be refused)")
    if ld > 0 and ldec > 0:
        parser.error(f"lambda_diverse={ld} and lambda_decor={ldec} are both "
                     f"active -- that hybrid is not a registered Gate A "
                     f"configuration (附录 G: 无 diversity / 平均余弦 / 去相关, "
                     f"exactly one family)")

    name = experiment_name(config)
    out_path = args.out or os.path.join("configs", f"{name}.yaml")

    with open(out_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)

    print(f"wrote {out_path}")
    print(f"experiment name: {name}")
    print(f"training: K={config['model'].get('num_concepts')} "
          f"seed={train.get('seed')} epochs={train['epochs']} "
          f"patience={train['patience']}")
    print(f"          lambda_compact={train.get('lambda_compact')} "
          f"lambda_diverse={ld} lambda_decor={ldec}")
    print("checks passed: patience >= epochs (fixed budget); "
          "at most one diversity family active")


if __name__ == "__main__":
    main()
