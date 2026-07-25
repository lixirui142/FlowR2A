#!/bin/bash

# Usage: bash scripts/caching/patch_reward.sh [CHUNK_ID] [TOTAL_CHUNKS]
# Example: bash scripts/caching/patch_reward.sh 0
#          bash scripts/caching/patch_reward.sh 1 4

CHUNK_ID=${1:-0}
TOTAL_CHUNKS=${2:-4}

# Wait until reward caching finishes before patching.
while pgrep -f run_train_reward_caching.py > /dev/null; do
    echo "waiting for run_train_reward_caching.py to finish..."
    sleep 60
done

python scripts/caching/patch_reward.py \
     --token-list-json bad_reward_files.json \
     --pdm-cache-dir $NAVSIM_EXP_ROOT/train_pdm_cache \
     --num-workers 64 \
     --skip-existing \
     --chunks ${CHUNK_ID}/${TOTAL_CHUNKS}
