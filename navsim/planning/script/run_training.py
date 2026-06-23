from typing import Tuple
from pathlib import Path
import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule


import os

import warnings
DEBUG = bool(int(os.environ["DEBUG"])) if "DEBUG" in os.environ else False
# import faulthandler
# faulthandler.enable()
# if DEBUG:
#     print("DEBUG on. Turn off cudnn")
#     import torch
#     torch.backends.cudnn.enabled = False
#     torch.backends.cudnn.benchmark = False # 暂时关闭 benchmark

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
# CONFIG_NAME = "default_training"
CONFIG_NAME = os.environ["TRAIN_CONFIG"] if "TRAIN_CONFIG" in os.environ else "default_training"
print("config name", CONFIG_NAME)


def custom_collate_fn(batch):
    """Collate a batch of (features, targets) dicts via default_collate."""
    features_list, targets_list = zip(*batch)
    features = default_collate(list(features_list))
    targets = default_collate(list(targets_list))
    return features, targets


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)
    reward_cache_path = getattr(cfg, "reward_cache_path", None)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
        reward_cache_path=reward_cache_path,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
        reward_cache_path=reward_cache_path,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent, training_epochs = cfg.trainer.params.max_epochs)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        reward_cache_path = getattr(cfg, "reward_cache_path", None)

        # Try loading pre-computed valid cache paths
        cache_path = Path(cfg.cache_path)
        train_precache = cache_path / "valid_cache_paths_train.pkl"
        val_precache = cache_path / "valid_cache_paths_val.pkl"
        train_valid, val_valid = None, None
        if train_precache.is_file() and val_precache.is_file():
            logger.info("Loading pre-cached valid paths from %s", cache_path)
            import pickle
            with open(train_precache, "rb") as f:
                train_rel = pickle.load(f)
            train_valid = {token: cache_path / rel for token, rel in train_rel.items()}
            with open(val_precache, "rb") as f:
                val_rel = pickle.load(f)
            val_valid = {token: cache_path / rel for token, rel in val_rel.items()}

        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
            reward_cache_path=reward_cache_path,
            valid_cache_paths=train_valid,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
            reward_cache_path=reward_cache_path,
            valid_cache_paths=val_valid,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True, collate_fn=custom_collate_fn)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False, collate_fn=custom_collate_fn)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())

    logger.info("Starting Training")
    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path:
        logger.info(f"Resuming from checkpoint: {ckpt_path}")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
      main()
