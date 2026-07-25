#!/bin/bash

# Usage: bash scripts/caching/run_train_reward_caching.sh [CHUNK_ID] [TOTAL_CHUNKS]
# Example: bash scripts/caching/run_train_reward_caching.sh 0
#          bash scripts/caching/run_train_reward_caching.sh 1 4

CHUNK_ID=${1:-0}
TOTAL_CHUNKS=${2:-4}

python scripts/caching/run_train_reward_caching.py \
     --training-cache-dir $NAVSIM_EXP_ROOT/training_cache_4lidar_gv_fix \
     --pdm-cache-dir $NAVSIM_EXP_ROOT/train_pdm_cache \
     --reward-pack-path /mnt/pfs-am/e2e-data/lixirui/navsim_workplace/dataset/traj_pdm_v2/ori/navtrain_8192_density_ep_normed.pkl \
     --num-workers 64 \
     --skip-existing \
     --chunks ${CHUNK_ID}/${TOTAL_CHUNKS}
