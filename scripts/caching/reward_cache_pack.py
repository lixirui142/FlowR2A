"""
Pack / unpack the per-token reward cache for distribution.

The reward cache lives as one `transfuser_reward.gz` per token inside the
training feature cache: `<training-cache>/<scene>/<token>/transfuser_reward.gz`.
There are ~100k of them, which is awkward to upload/download. This script

  - `aggregate`: walks the training cache, loads every `transfuser_reward.gz`
    into a single {token: reward_dict} mapping, and splits it into fixed-size
    gzipped part files (`reward_cache_part_000.pkl.gz`, ...) for upload to Hugging Face.

  - `extract`: loads the part files and writes each token's `transfuser_reward.gz`
    back into the training cache, matching tokens to their scene dir by walking
    the cache (same layout run_dataset_caching.py produced).

Usage:
    # pack into ~4000-token parts
    python scripts/caching/reward_cache_pack.py aggregate \
        --training-cache-dir $NAVSIM_EXP_ROOT/training_cache \
        --out-dir $NAVSIM_EXP_ROOT/reward_cache_pack \
        --tokens-per-part 4000 --num-workers 64

    # unpack downloaded parts back into the cache
    python scripts/caching/reward_cache_pack.py extract \
        --pack-dir $NAVSIM_EXP_ROOT/reward_cache_pack \
        --training-cache-dir $NAVSIM_EXP_ROOT/training_cache \
        --num-workers 64 --skip-existing
"""

import argparse
import glob
import gzip
import os
import pickle
import sys
from pathlib import Path
from multiprocessing import Pool

from tqdm import tqdm

# Add project root to path (scripts/caching/ -> repo root is two levels up)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from navsim.planning.training.dataset import (  # noqa: E402
    load_feature_target_from_pickle,
    dump_feature_target_to_pickle,
)

_PART_GLOB = "reward_cache_part_*.pkl.gz"


def _build_token_list(training_cache_dir):
    """Walk training cache -> list of (token, token_dir). Reward file may or may not exist."""
    token_list = []
    for scene_path in sorted(Path(training_cache_dir).iterdir()):
        if not scene_path.is_dir():
            continue
        for token_path in sorted(scene_path.iterdir()):
            if token_path.is_dir():
                token_list.append((token_path.name, str(token_path)))
    return token_list


# ---------------------------------------------------------------- aggregate

def _load_one(args):
    token, token_dir = args
    path = os.path.join(token_dir, "transfuser_reward.gz")
    if not os.path.exists(path):
        return None
    try:
        return token, load_feature_target_from_pickle(Path(path))
    except Exception as e:
        print(f"Warning: failed to load {path}: {e}")
        return None


def aggregate(args):
    # Token order is deterministic (sorted walk), so batch i always maps to the
    # same part file -> the run is resumable by skipping existing parts. Each
    # batch is loaded with a bounded pool.map that completes before the next,
    # keeping peak memory to ~one part rather than buffering the whole cache.
    token_list = _build_token_list(args.training_cache_dir)
    print(f"Found {len(token_list)} token dirs in training cache")
    os.makedirs(args.out_dir, exist_ok=True)

    n = args.tokens_per_part
    n_parts = (len(token_list) + n - 1) // n
    n_loaded = 0
    with Pool(args.num_workers) as p:
        for part_idx in range(n_parts):
            out = os.path.join(args.out_dir, f"reward_cache_part_{part_idx:03d}.pkl.gz")
            batch = token_list[part_idx * n:(part_idx + 1) * n]
            if args.skip_existing and os.path.exists(out):
                print(f"  skip {os.path.basename(out)} (exists)")
                continue
            buf = {}
            for r in tqdm(p.imap(_load_one, batch, chunksize=16),
                          total=len(batch), desc=f"part {part_idx:03d}/{n_parts - 1}"):
                if r is not None:
                    buf[r[0]] = r[1]
            with gzip.open(out, "wb", compresslevel=1) as f:
                pickle.dump(buf, f, protocol=pickle.HIGHEST_PROTOCOL)
            n_loaded += len(buf)
            print(f"  wrote {out} ({len(buf)} tokens, {os.path.getsize(out) / 1e9:.2f} GB)")
    print(f"Done. Aggregated {n_loaded} reward caches into ~{n_parts} part(s) at {args.out_dir}")


# ------------------------------------------------------------------ extract
# One part is held in the main process at a time (parts can total 100s of GB);
# each reward_dict is shipped to a worker for the gzip write.

def _write_one(args):
    token_dir, reward_dict, skip_existing = args
    out = os.path.join(token_dir, "transfuser_reward.gz")
    if skip_existing and os.path.exists(out):
        return "skipped"
    dump_feature_target_to_pickle(Path(out), reward_dict)
    return "written"


def extract(args):
    parts = sorted(glob.glob(os.path.join(args.pack_dir, _PART_GLOB)))
    if not parts:
        sys.exit(f"No {_PART_GLOB} files found in {args.pack_dir}")
    print(f"Found {len(parts)} part file(s) in {args.pack_dir}")

    token_dirs = dict(_build_token_list(args.training_cache_dir))  # token -> token_dir
    print(f"Found {len(token_dirs)} token dirs in training cache")

    counts = {"written": 0, "skipped": 0, "missing": 0}
    with Pool(args.num_workers) as p:
        for part in parts:
            with gzip.open(part, "rb") as f:
                reward_index = pickle.load(f)
            work = [(token_dirs[t], rd, args.skip_existing)
                    for t, rd in reward_index.items() if t in token_dirs]
            counts["missing"] += len(reward_index) - len(work)
            for r in tqdm(p.imap_unordered(_write_one, work, chunksize=16),
                          total=len(work), desc=f"Extracting {os.path.basename(part)}"):
                counts[r] += 1

    print(f"Done. written={counts['written']} skipped={counts['skipped']} "
          f"not-in-cache={counts['missing']}")
    if counts["missing"]:
        print("  (not-in-cache tokens exist in the pack but have no matching token dir; "
              "your feature cache may cover a different split)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("aggregate", help="pack per-token reward caches into part files")
    a.add_argument("--training-cache-dir", required=True)
    a.add_argument("--out-dir", required=True)
    a.add_argument("--tokens-per-part", type=int, default=4000,
                   help="tokens per part file (default: 4000)")
    a.add_argument("--num-workers", type=int, default=64)
    a.add_argument("--skip-existing", action="store_true",
                   help="skip part files that already exist (resume a killed run)")
    a.set_defaults(func=aggregate)

    e = sub.add_parser("extract", help="unpack part files back into the training cache")
    e.add_argument("--pack-dir", required=True)
    e.add_argument("--training-cache-dir", required=True)
    e.add_argument("--num-workers", type=int, default=64)
    e.add_argument("--skip-existing", action="store_true",
                   help="skip tokens that already have transfuser_reward.gz")
    e.set_defaults(func=extract)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
