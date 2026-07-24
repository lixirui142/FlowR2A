#!/bin/bash
# Stage 2: scorer-only training, initialized from a stage-1 checkpoint.
# Set S1_CKPT to your trained stage-1 weights below.
set -e

export TRAIN_CONFIG=flowr2a_training_s2

S1_CKPT="/mnt/pfs-am/e2e-data/lixirui/navsim_workplace/exp/flowr2a_agent_s1/lightning_logs/version_0/checkpoints/last.ckpt"

bash scripts/training/run_single_train.sh flowr2a_agent_s2 \
    agent.config.checkpoint_path="'$S1_CKPT'"
