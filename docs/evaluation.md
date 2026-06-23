# Evaluation

This guide downloads the pre-trained FlowR2A checkpoints and evaluates them on the NAVSIM **navtest** split. Make sure you have finished [Data and Environment Preparation](install.md) first.

## 1. Download checkpoints
The pre-trained checkpoint is hosted on the [🤗 Hugging Face model page](https://huggingface.co/lixirui142/FlowR2A). Download it into `ckpts/`:

```bash
mkdir -p ckpts
# Option A: direct download
wget -O ckpts/flowr2a_s2.ckpt https://huggingface.co/lixirui142/FlowR2A/resolve/main/flowr2a_s2.ckpt

# Option B: via the Hugging Face CLI (pip install -U "huggingface_hub[cli]")
huggingface-cli download lixirui142/FlowR2A flowr2a_s2.ckpt --local-dir ckpts
```

## 2. Cache the evaluation data
Pre-compute the metric and feature caches for navtest (one-time step, speeds up evaluation):
```bash
# Metric cache
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
    train_test_split=navtest \
    cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache

# Feature cache
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
    agent=flowr2a_agent_s2 \
    experiment_name=flowr2a_cache \
    train_test_split=navtest \
    cache_path=$NAVSIM_EXP_ROOT/testing_cache
```

## 3. Run evaluation
Use the provided launcher, which wraps `run_pdm_score_fast.py` with the inference parameters aligned to `navsim/agents/flowr2a/flowr2a_config.py`:
```bash
python scripts/evaluation/run_eval.py \
    --exp_id flowr2a_agent_s2 \
    --checkpoint ckpts/flowr2a_s2.ckpt
```

By default the feature cache at `$NAVSIM_EXP_ROOT/testing_cache` is used; override it with `--feature_cache_path`. Inference parameters (reward target, number of sampled trajectories, guidance weights, flow steps) can be tuned via flags — run `python scripts/evaluation/run_eval.py --help` for the full list.

The script reports the PDMS score on navtest when it finishes.
