"""
Save predicted trajectories to a pickle file for evaluation with navsim-v2.

This script runs the inference phase of the agent (batched, multi-GPU via accelerate)
and saves the best trajectory per scene token. The output pickle can be used directly
with navsim-v2's evaluation scripts.

Output format:
  {token: poses_array}  where poses_array is np.float32 of shape (8, 3) in ego-local frame.

Usage:
  accelerate launch navsim/planning/script/run_save_trajectories.py \
      train_test_split=navtest \
      agent=<agent_config> \
      agent.checkpoint_path='<path>' \
      +feature_cache_path=<path> \
      +output_trajectory_path=<path_to_save.pkl>
"""

from typing import Any, Dict
from pathlib import Path
import logging
import pickle
import os
import warnings

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator

from nuplan.planning.script.builders.logging_builder import build_logger
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneFilter
from navsim.planning.training.dataset import CacheOnlyDataset

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def batch_to_device(batch: Any, device: torch.device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    elif isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [batch_to_device(v, device) for v in batch]
    else:
        return batch


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    accelerator = Accelerator()
    if accelerator.is_main_process:
        build_logger(cfg)
        logger.info(f"Accelerator initialized. Device count: {accelerator.num_processes}")

    # 1. Initialize Agent
    agent: AbstractAgent = hydra.utils.instantiate(cfg.agent)
    agent.initialize()

    # 2. Setup Data Loading
    scene_filter = hydra.utils.instantiate(cfg.train_test_split.scene_filter)

    test_dataset = CacheOnlyDataset(
        cache_path=cfg.feature_cache_path,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        log_names=scene_filter.log_names,
        train=False,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.get("inference_batch_size", 64),
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

    agent, test_dataloader = accelerator.prepare(agent, test_dataloader)
    if accelerator.is_main_process:
        logger.info(f"Starting inference on {len(test_dataset)} frames...")

    # 3. Batch Inference
    global_trajectories: Dict[str, np.ndarray] = {}

    agent.eval()
    with torch.no_grad():
        for batch in tqdm(
            test_dataloader,
            desc="Inference",
            disable=not accelerator.is_local_main_process,
        ):
            features, targets = batch
            batch_return = agent.forward(features, targets, multi_return=True)

            batch_full_trajs = batch_return["trajs"]       # (B, num_traj, 8, 3)
            batch_best_idx = batch_return["best_idx"]      # (B,)

            # Gather across GPUs
            full_trajs = accelerator.gather_for_metrics(batch_full_trajs)
            best_idxs = accelerator.gather_for_metrics(batch_best_idx)
            batch_tokens = features["token"]
            all_tokens = accelerator.gather_for_metrics(batch_tokens)

            if accelerator.is_main_process:
                full_trajs_cpu = full_trajs.detach().cpu().numpy()
                best_idxs_cpu = best_idxs.cpu().numpy()

                for i, token in enumerate(all_tokens):
                    best_idx = best_idxs_cpu[i]
                    # Select best trajectory: (8, 3) in ego-local frame, float32
                    best_traj = full_trajs_cpu[i, best_idx].astype(np.float32)
                    global_trajectories[token] = best_traj

    accelerator.wait_for_everyone()

    if not accelerator.is_main_process:
        return

    # 4. Save
    output_path = cfg.get("output_trajectory_path", None)
    if output_path is None:
        # Default: save next to checkpoint
        ckpt_dir = Path(os.path.dirname(cfg.agent.checkpoint_path))
        result_suffix = cfg.get("result_suffix", "")
        output_path = ckpt_dir / f"trajectories{result_suffix}.pkl"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(global_trajectories, f)

    logger.info(
        f"Saved {len(global_trajectories)} trajectories to {output_path}"
    )


if __name__ == "__main__":
    main()
