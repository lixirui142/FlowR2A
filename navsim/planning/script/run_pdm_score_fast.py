from typing import Any, Dict, List
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import traceback
import logging
import lzma
import pickle
import os
import uuid

import torch
import numpy as np
from tqdm import tqdm

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd
from torch.utils.data import DataLoader

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.training.dataset import CacheOnlyDataset
import warnings
from accelerate import Accelerator

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def run_pdm_score(args):
    """CPU worker: PDM-score pre-computed trajectories (no agent / inference here).

    Each task carries the agent's sampled trajectories for a token plus its
    metric_cache path. With ``eval_all_trajs`` every sampled trajectory is scored
    (full resolution on all generated trajs); otherwise only the agent-selected
    ``best_idx`` is scored. Both modes report the same headline row (best_idx).
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    if not args:
        return []

    cfg: DictConfig = args[0]["cfg"]
    eval_all_trajs = args[0].get("eval_all_trajs", False)
    save_full_results = args[0].get("save_full_results", False)
    items = sum([split["items"] for split in args], start=[])

    # Instantiate Simulator and Scorer once per worker
    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"

    pdm_results: List[Dict[str, Any]] = []

    for idx, item in enumerate(items):
        token = item["token"]
        trajs = item["trajectory"]["traj"]
        best_idx = item["trajectory"]["best_idx"]
        metric_cache_path = item["metric_cache_path"]

        score_row: Dict[str, Any] = {"token": token, "valid": True}

        try:
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)

            def _score(traj):
                return asdict(
                    pdm_score(
                        metric_cache=metric_cache,
                        model_trajectory=Trajectory(traj),
                        future_sampling=simulator.proposal_sampling,
                        simulator=simulator,
                        scorer=scorer,
                    )
                )

            if eval_all_trajs:
                traj_score_list = [_score(traj) for traj in trajs]
                score_row.update(traj_score_list[best_idx])
            else:
                traj_score_list = [_score(trajs[best_idx])]
                score_row.update(traj_score_list[0])

            if save_full_results:
                # Append the raw trajectory dict so full_results.pkl carries everything.
                traj_score_list.append(item["trajectory"])
                score_row["traj_score_list"] = traj_score_list

        except Exception:
            logger.warning(f"----------- Evaluation failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False

        pdm_results.append(score_row)

        if (idx + 1) % 10 == 0:
            logger.info(f"Processed {idx + 1}/{len(items)} scenarios in thread {thread_id}")

    return pdm_results


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """PDMS evaluation: batched GPU inference -> parallel CPU PDM scoring -> CSV.

    cfg switches:
      - eval_all_trajs (default False): score every sampled trajectory vs only the
        agent-selected best. Both give the same headline score; "all" adds full
        per-trajectory resolution.
      - save_full_results (default False): dump full_results.pkl with per-trajectory
        scores + subscores. Off = evaluation only.
    """
    eval_all_trajs = cfg.get("eval_all_trajs", False)
    save_full_results = cfg.get("save_full_results", False)

    accelerator = Accelerator()
    if accelerator.is_main_process:
        build_logger(cfg)
        logger.info(f"Accelerator initialized. Device count: {accelerator.num_processes}")
        logger.info("Initializing Agent and Data Loaders...")

    # 1. Initialize Agent
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    # 2. Setup Data Loading
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    test_dataset = CacheOnlyDataset(
        cache_path=cfg.feature_cache_path,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        log_names=scene_filter.log_names,
        train=False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

    agent, test_dataloader = accelerator.prepare(agent, test_dataloader)
    if accelerator.is_main_process:
        logger.info(f"Starting Inference on {len(test_dataset)} frames...")

    # 3. Batch inference loop (gather sampled trajectories + best_idx per token)
    global_trajectories: Dict[str, Any] = {}
    agent.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Inference", disable=not accelerator.is_local_main_process):
            features, targets = batch
            batch_return = agent.forward(features, targets, multi_return=True)

            best_idxs = accelerator.gather_for_metrics(batch_return["best_idx"])
            full_trajs = accelerator.gather_for_metrics(batch_return["trajs"])
            all_tokens = accelerator.gather_for_metrics(features["token"])

            gathered_subscore = {}
            if save_full_results:
                gathered_subscore = {
                    k: accelerator.gather_for_metrics(v) for k, v in batch_return.get("subscore", {}).items()
                }

            if accelerator.is_main_process:
                full_trajs_cpu = full_trajs.detach().cpu().numpy()
                best_idxs_cpu = best_idxs.cpu().numpy()
                subscore_cpu = {k: v.cpu().numpy() for k, v in gathered_subscore.items()}

                for i, token in enumerate(all_tokens):
                    if token is None:
                        continue  # skip padding slots
                    entry = {"traj": full_trajs_cpu[i], "best_idx": best_idxs_cpu[i]}
                    if save_full_results:
                        entry["meta_infos"] = batch_return.get("meta_infos", {})
                        entry["subscore"] = {k: v[i] for k, v in subscore_cpu.items()}
                    global_trajectories[token] = entry

    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    logger.info("Inference complete. Starting PDM Scoring...")

    # Sanity check: all expected tokens populated
    missing = set(test_dataset.tokens) - set(global_trajectories.keys())
    if missing:
        logger.warning(f"Missing {len(missing)} tokens in global_trajectories (e.g. {list(missing)[:5]})")

    # 4. Scoring (parallel CPU workers over tokens with a metric cache)
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    tokens_to_evaluate = list(set(global_trajectories.keys()) & set(metric_cache_loader.tokens))
    if len(tokens_to_evaluate) < len(global_trajectories):
        logger.warning(
            f"Skipping {len(global_trajectories) - len(tokens_to_evaluate)} tokens due to missing MetricCache."
        )

    all_tasks = [
        {
            "token": token,
            "trajectory": global_trajectories[token],
            "metric_cache_path": metric_cache_loader.metric_cache_paths[token],
        }
        for token in tokens_to_evaluate
    ]

    chunk_size = 20
    chunked_tasks = [
        {
            "cfg": cfg,
            "items": all_tasks[i : i + chunk_size],
            "eval_all_trajs": eval_all_trajs,
            "save_full_results": save_full_results,
        }
        for i in range(0, len(all_tasks), chunk_size)
    ]

    worker = build_worker(cfg)
    logger.info(f"Distributing {len(all_tasks)} tasks across workers...")
    score_rows = worker_map(worker, run_pdm_score, chunked_tasks)

    # 5. Optionally save full per-trajectory results, then aggregate to CSV
    if save_full_results:
        full_results_path = Path(os.path.dirname(cfg.agent.checkpoint_path)) / "full_results.pkl"
        with open(full_results_path, "wb") as f:
            pickle.dump(score_rows, f)
        logger.info(f"Saved full results to {full_results_path}")

    for row in score_rows:
        row.pop("traj_score_list", None)

    pdm_score_df = pd.DataFrame(score_rows)
    if pdm_score_df.empty:
        logger.error("No results generated.")
        return

    num_successful = pdm_score_df["valid"].sum()
    num_failed = len(pdm_score_df) - num_successful

    average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    save_path = Path(cfg.output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    csv_path = save_path / f"{timestamp}.csv"
    pdm_score_df.to_csv(csv_path)

    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_successful}.
            Number of failed scenarios: {num_failed}.
            Final average score of valid results: {pdm_score_df['score'].mean()}.
            Final average ttc of valid results: {pdm_score_df['time_to_collision_within_bound'].mean()}.
            Final average ep of valid results: {pdm_score_df['ego_progress'].mean()}.
            Results are stored in: {csv_path}.
        """
    )


if __name__ == "__main__":
    main()
