"""Lightweight dummy agent for dataset caching with v4 features (4-frame lidar + 1-frame camera)."""

from typing import Dict, List

import torch

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.flowr2a.flowr2a_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder


class CachingAgent(AbstractAgent):
    """Dummy agent for caching v4 features: 1-frame camera + 4-frame lidar."""

    def __init__(self, config, lr: float = 0.0, checkpoint_path=None, **kwargs):
        super().__init__()
        self._config = config

    def name(self) -> str:
        return "CachingAgent"

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig(
            cam_f0=[3],
            cam_l0=[3],
            cam_l1=[],
            cam_l2=[],
            cam_r0=[3],
            cam_r1=[],
            cam_r2=[],
            cam_b0=[],
            lidar_pc=[0, 1, 2, 3],
        )

    def initialize(self) -> None:
        pass

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [TransfuserFeatureBuilder(config=self._config)]

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [TransfuserTargetBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("CachingAgent does not support forward pass.")

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        raise NotImplementedError("CachingAgent does not support inference.")
