#!/bin/bash
# Stage 2: scorer-only training, initialized from a stage-1 checkpoint.
# Pass the stage-1 checkpoint path as the first argument.
set -e

export TRAIN_CONFIG=flowr2a_training_s2

S1_CKPT=$1

bash scripts/training/run_single_train.sh flowr2a_agent_s2 \
    agent.config.checkpoint_path="'$S1_CKPT'"
