#!/bin/bash
# Train a flowr2a agent on navtrain.
#
#   bash scripts/training/run_single_train.sh flowr2a_agent_s1        # stage 1: flow decoder
#   bash scripts/training/run_single_train.sh flowr2a_agent_s2        # stage 2: scorer-only
#   bash scripts/training/run_single_train.sh flowr2a_agent_s1 <ckpt> # resume from checkpoint
#
# Stage 2 loads the stage-1 weights via `config.checkpoint_path` set in
# navsim/planning/script/config/common/agent/flowr2a_agent_s2.yaml.
set -e

EXP_ID=$1        # agent config name, e.g. flowr2a_agent_s1
CKPT_PATH=$2     # optional: checkpoint to resume training from

if [ -z "$EXP_ID" ]; then
    echo "Usage: bash run_single_train.sh <agent_config> [resume_ckpt]"
    exit 1
fi

RESUME_ARG=""
if [ -n "$CKPT_PATH" ]; then
    echo "Resuming $EXP_ID from checkpoint: $CKPT_PATH"
    RESUME_ARG="+ckpt_path='$CKPT_PATH'"
else
    echo "Starting training: $EXP_ID"
fi

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    agent=$EXP_ID \
    experiment_name=$EXP_ID \
    train_test_split=navtrain \
    split=trainval \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    $RESUME_ARG

# Evaluate the trained checkpoint with:
#   python scripts/evaluation/run_eval.py --exp_id $EXP_ID --checkpoint <ckpt>
