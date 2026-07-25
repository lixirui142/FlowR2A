"""
Patch existing transfuser_reward.gz in-place: add ttc_time_b15.

Re-runs PDM simulation at ttc_bound=1.5 (use_history_comfort=True,
driving_direction_exclude_intersection=True) and saves the resulting per-step
ttc_time as ttc_time_b15. Nothing else in the reward dict is touched.

Usage:
    python scripts/caching/patch_reward.py \
        --training-cache-dir $NAVSIM_EXP_ROOT/training_cache_4lidar_gv \
        --pdm-cache-dir $NAVSIM_EXP_ROOT/train_pdm_cache \
        --num-workers 32 --skip-existing --chunks 0/4
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add project root to path (scripts/caching/ -> repo root is two levels up)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from navsim.planning.training.dataset import (
    load_feature_target_from_pickle,
    dump_feature_target_to_pickle,
)
from navsim.agents.flowr2a.pdm_scoring import _pdm_worker, _init_pool

_TTC_BOUND = 1.5
_USE_HISTORY_COMFORT = True
_DDC_EXCLUDE_INTERSECTION = True


def _load_scoring_config():
    """Load default PDM scoring config (simulator + scorer) from YAML."""
    from omegaconf import OmegaConf

    config_path = (
        _PROJECT_ROOT / "navsim" / "planning" / "script" / "config"
        / "pdm_scoring" / "default_scoring_parameters.yaml"
    )
    cfg = OmegaConf.load(config_path)
    return cfg.simulator, cfg.scorer


def _build_token_list(training_cache_dir):
    """Walk training cache to build list of (token, token_dir, scene_dir)."""
    token_list = []
    training_cache = Path(training_cache_dir)
    for scene_path in sorted(training_cache.iterdir()):
        if not scene_path.is_dir():
            continue
        scene_dir = scene_path.name
        for token_path in sorted(scene_path.iterdir()):
            if not token_path.is_dir():
                continue
            token = token_path.name
            token_list.append((token, str(token_path), scene_dir))
    return token_list


def _token_list_from_json(json_path):
    """Build (token, token_dir, scene_dir) from a json list of reward .gz paths."""
    paths = json.load(open(json_path))
    token_list = []
    for p in paths:
        token_dir = os.path.dirname(p)  # .../scene_dir/token
        token = os.path.basename(token_dir)
        scene_dir = os.path.basename(os.path.dirname(token_dir))
        token_list.append((token, token_dir, scene_dir))
    return token_list


def _process_token(args):
    """Worker: add ttc_time_b15 to one existing transfuser_reward.gz in-place."""
    token, token_dir, scene_dir, pdm_cache_dir, traj_8step, skip_existing = args

    reward_path = os.path.join(token_dir, "transfuser_reward.gz")

    try:
        reward_dict = load_feature_target_from_pickle(Path(reward_path))
    except Exception as e:
        print(f"Warning: failed to load {reward_path}: {e}")
        return token

    if skip_existing and "ttc_time_b15" in reward_dict:
        return token

    metric_cache_path = os.path.join(
        pdm_cache_dir, scene_dir, "unknown", token, "metric_cache.pkl"
    )
    if not os.path.exists(metric_cache_path):
        print(f"Warning: no metric cache for {token} ({scene_dir}), skipping")
        return token

    try:
        _scores, subscores = _pdm_worker((metric_cache_path, traj_8step))
    except Exception as e:
        print(f"Warning: PDM sim failed for {token} ({scene_dir}): {e}")
        return token

    reward_dict["ttc_time_b15"] = subscores["ttc_time"]  # (8192, 40)

    dump_feature_target_to_pickle(Path(reward_path), reward_dict)
    return token


def main():
    parser = argparse.ArgumentParser(
        description="Patch transfuser_reward.gz: add ttc_time_b15 (ttc_bound=1.5)"
    )
    parser.add_argument("--training-cache-dir", default=None,
                        help="Path to training cache (has scene_dir/token/ subdirs). "
                             "Required unless --token-list-json is given.")
    parser.add_argument("--token-list-json", default=None,
                        help="Optional json list of reward .gz paths to patch "
                             "(e.g. bad_reward_files.json). Skips walking the cache dir.")
    parser.add_argument("--pdm-cache-dir", required=True,
                        help="Path to PDM metric cache (has scene_dir/unknown/token/ subdirs)")
    parser.add_argument("--traj-list-path",
                        default="/mnt/pfs-am/e2e-data/lixirui/navsim_workplace/gtrs/traj_final/8192.npy",
                        help="Path to vocab trajectories .npy (default: 8192.npy)")
    parser.add_argument("--num-workers", type=int, default=32,
                        help="Number of parallel workers (default: 32)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip tokens that already have ttc_time_b15")
    parser.add_argument("--chunks", type=str, default=None,
                        help="Process chunk i of N, format: i/N (e.g. 0/4)")
    args = parser.parse_args()

    # 1. Load vocab trajectories and downsample to 8 steps
    print("Loading vocab trajectories...")
    traj_raw = np.load(args.traj_list_path)  # (8192, 40, 3)
    traj_8step = traj_raw[:, 4::5, :]        # (8192, 8, 3)
    print(f"  Raw shape: {traj_raw.shape}, downsampled: {traj_8step.shape}")

    # 2. Build token list (from json if given, else walk the cache dir)
    print("Building token list...")
    if args.token_list_json:
        token_list = _token_list_from_json(args.token_list_json)
        print(f"  Loaded {len(token_list)} tokens from {args.token_list_json}")
    elif args.training_cache_dir:
        token_list = _build_token_list(args.training_cache_dir)
        print(f"  Found {len(token_list)} tokens in training cache")
    else:
        parser.error("either --token-list-json or --training-cache-dir is required")

    # 3. Filter: only tokens with existing transfuser_reward.gz
    filtered = []
    skipped_no_reward = 0
    for token, token_dir, scene_dir in token_list:
        reward_path = os.path.join(token_dir, "transfuser_reward.gz")
        if not os.path.exists(reward_path):
            skipped_no_reward += 1
            continue
        filtered.append((token, token_dir, scene_dir))

    print(f"  Skipped {skipped_no_reward} tokens (no existing reward cache)")
    if args.skip_existing:
        print("  --skip-existing: workers will skip tokens that already have ttc_time_b15")
    print(f"  Remaining: {len(filtered)} tokens to process")

    # 4. Apply chunking
    if args.chunks:
        chunk_idx, num_chunks = map(int, args.chunks.split("/"))
        chunk_size = (len(filtered) + num_chunks - 1) // num_chunks
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(filtered))
        filtered = filtered[start:end]
        print(f"  Chunk {chunk_idx}/{num_chunks}: processing {len(filtered)} tokens [{start}:{end}]")

    if not filtered:
        print("Nothing to process. Done.")
        return

    # 5. Prepare worker arguments
    worker_args = [
        (token, token_dir, scene_dir, args.pdm_cache_dir, traj_8step, args.skip_existing)
        for token, token_dir, scene_dir in filtered
    ]

    # 6. Launch pool
    print(f"Launching pool with {args.num_workers} workers...")
    print(f"  ttc_bound={_TTC_BOUND}, use_history_comfort={_USE_HISTORY_COMFORT}, "
          f"ddc_exclude_intersection={_DDC_EXCLUDE_INTERSECTION}")
    sim_cfg, scorer_cfg = _load_scoring_config()

    with mp.Pool(
        args.num_workers,
        initializer=_init_pool,
        initargs=(
            sim_cfg,
            scorer_cfg,
            _TTC_BOUND,
            _USE_HISTORY_COMFORT,
            _DDC_EXCLUDE_INTERSECTION,
        ),
    ) as pool:
        with tqdm(total=len(worker_args), desc="Patching ttc_time_b15") as pbar:
            for _token in pool.imap_unordered(_process_token, worker_args):
                pbar.update(1)

    print("Done!")


if __name__ == "__main__":
    main()
