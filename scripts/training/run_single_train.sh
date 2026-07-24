#!/bin/bash
# Train a flowr2a agent on navtrain. Any extra args are passed through to
# run_training.py as Hydra overrides, e.g.:
#   bash scripts/training/run_single_train.sh flowr2a_agent_s2 \
#       agent.config.checkpoint_path='/path/to/s1.ckpt' +ckpt_path='/path/to/resume.ckpt'
set -e

EXP_ID=$1        # agent config name, e.g. flowr2a_agent_s1
shift || true    # remaining args are Hydra overrides

if [ -z "$EXP_ID" ]; then
    echo "Usage: bash run_single_train.sh <agent_config> [hydra_overrides...]"
    exit 1
fi

echo "Starting training: $EXP_ID"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    agent=$EXP_ID \
    experiment_name=$EXP_ID \
    train_test_split=navtrain \
    split=trainval \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    "$@"

# Evaluate the trained checkpoint with:
#   python scripts/evaluation/run_eval.py --exp_id $EXP_ID --checkpoint <ckpt>
