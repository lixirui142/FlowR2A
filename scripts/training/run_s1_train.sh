#!/bin/bash
# Stage 1: train the flowr2a flow decoder on navtrain.
set -e

export TRAIN_CONFIG=flowr2a_training_s1
bash scripts/training/run_single_train.sh flowr2a_agent_s1
