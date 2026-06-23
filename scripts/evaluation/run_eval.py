#!/usr/bin/env python3
"""Evaluate a released flowr2a model on navtest (single PDMS run).

Launches navsim/planning/script/run_pdm_score_fast.py via accelerate. Inference
parameters mirror navsim/agents/flowr2a/flowr2a_config.py; defaults below are kept
in sync with that config and only emitted as Hydra overrides when changed.

Example:
    python scripts/evaluation/run_eval.py \
        --exp_id flowr2a_agent_s2 \
        --checkpoint /path/to/epoch=1-step=666.ckpt
"""
import argparse
import os
import subprocess
import sys

# Inference defaults, aligned with navsim/agents/flowr2a/flowr2a_config.py
DEFAULTS = {
    "test_score_min": 0.9,
    "test_score_max": 1.0,
    "test_num_traj_sampling": 60,
    "test_weight_ttc": 1.0,
    "test_weight_ep": 1.0,
    "test_weight_c": 1.0,
    "init_step_min": 10,
    "init_step_max": 18,
}

# Feature cache.
DEFAULT_FEATURE_CACHE = os.path.join(
    os.environ.get("NAVSIM_EXP_ROOT", ""), "testing_cache"
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a flowr2a model on navtest")
    parser.add_argument("--exp_id", required=True, help="Agent config name, e.g. flowr2a_agent_s2")
    parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint (.ckpt)")
    parser.add_argument("--feature_cache_path", default=DEFAULT_FEATURE_CACHE,
                        help="Feature cache dir (default: $NAVSIM_EXP_ROOT/testing_cache)")
    parser.add_argument("--eval_all_trajs", action="store_true",
                        help="Score every sampled trajectory (default: only the agent-selected best)")
    parser.add_argument("--save_full_results", action="store_true",
                        help="Dump full_results.pkl with per-trajectory scores")
    # Inference parameters (see flowr2a_config.py)
    parser.add_argument("--test_score_min", type=float, default=DEFAULTS["test_score_min"])
    parser.add_argument("--test_score_max", type=float, default=DEFAULTS["test_score_max"])
    parser.add_argument("--test_num_traj_sampling", type=int, default=DEFAULTS["test_num_traj_sampling"])
    parser.add_argument("--test_weight_ttc", type=float, default=DEFAULTS["test_weight_ttc"])
    parser.add_argument("--test_weight_ep", type=float, default=DEFAULTS["test_weight_ep"])
    parser.add_argument("--test_weight_c", type=float, default=DEFAULTS["test_weight_c"])
    parser.add_argument("--init_step_min", type=int, default=DEFAULTS["init_step_min"])
    parser.add_argument("--init_step_max", type=int, default=DEFAULTS["init_step_max"])
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        sys.exit(f"Error: checkpoint not found: {args.checkpoint}")

    # Emit a Hydra override only for params that differ from the config defaults.
    overrides = [
        f"+agent.config.{k}={getattr(args, k)}"
        for k in DEFAULTS
        if getattr(args, k) != DEFAULTS[k]
    ]
    if args.eval_all_trajs:
        overrides.append("+eval_all_trajs=true")
    if args.save_full_results:
        overrides.append("+save_full_results=true")

    navsim_root = os.environ.get("NAVSIM_DEVKIT_ROOT", "")
    # Switch to accelerate launch for multi-gpu inference.
    cmd = [
        # "accelerate", "launch",
        "python",
        f"{navsim_root}/navsim/planning/script/run_pdm_score_fast.py",
        "train_test_split=navtest",
        f"agent={args.exp_id}",
        "worker=ray_distributed",
        f"agent.checkpoint_path='{args.checkpoint}'",
        f"experiment_name={args.exp_id}_eval",
        f"+feature_cache_path={args.feature_cache_path}",
    ] + overrides

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Command: {' '.join(cmd)}")

    env = os.environ.copy()

    env["EVAL"] = "1"
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
