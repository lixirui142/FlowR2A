# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is the **release-ready** open-source version of FlowR2A. Keep changes clean and minimal; prefer clarity over experimental sprawl.

## Project Overview

FlowR2A is a multimodal driving planner that learns the reward-conditioned action distribution *p(a | r)* with flow matching. Simulation rewards are used as a *condition* rather than a discriminative target; at inference, generation is steered toward high-reward trajectories via classifier-free guidance. See @README.md for the full description.

Built on NAVSIM, using PyTorch Lightning + Hydra.

## Key Locations

- **Agent code**: `navsim/agents/flowr2a/` — model (`flowr2a_model.py`), config (`flowr2a_config.py`), agent entry (`flowr2a_agent.py`).
- **Agent configs (Hydra)**: `navsim/planning/script/config/common/agent/` — `agent=flowr2a_agent_s1` (stage 1, flow decoder) and `agent=flowr2a_agent_s2` (stage 2, scorer-only) resolve to the YAMLs here.
- **Entry points**: `navsim/planning/script/run_training.py`, `run_pdm_score_fast.py`.
- **Scripts**: `scripts/training/run_single_train.sh`, `scripts/evaluation/run_eval.py`.
- **Docs**: `docs/install.md` (env + data setup), `docs/evaluation.md` (run eval).

## Common Commands

```bash
# Train (stage 1, then stage 2)
bash scripts/training/run_single_train.sh flowr2a_agent_s1
bash scripts/training/run_single_train.sh flowr2a_agent_s2   # set config.checkpoint_path to the stage-1 ckpt first

# Evaluate on navtest
python scripts/evaluation/run_eval.py --exp_id flowr2a_agent_s2 --checkpoint /path/to/epoch=...ckpt
```

## Environment

Requires `NAVSIM_DEVKIT_ROOT` (repo root) and `NAVSIM_EXP_ROOT` (experiment/cache output). See `docs/install.md` for full setup.
