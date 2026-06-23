from typing import Any, List, Dict, Optional

import torch
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.flowr2a.flowr2a_config import TransfuserConfig
from navsim.agents.flowr2a.flowr2a_callback import TransfuserCallback
from navsim.agents.flowr2a.flowr2a_loss import transfuser_loss
from navsim.agents.flowr2a.modules.scheduler import WarmupCosLR
from navsim.agents.flowr2a.flowr2a_model import V2TransfuserModel
from navsim.agents.flowr2a.flowr2a_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder


class TransfuserAgent(AbstractAgent):
    """Reward-conditioned flow-based trajectory decoder agent (ResNet34 backbone).

    A single agent class covers both training stages; the stage is selected by config:
      - Stage 1: joint decoder + scorer training (``scorer_only`` unset/False).
      - Stage 2: scorer-only finetune on the frozen decoder's output
        (``scorer_only=True`` with ``checkpoint_path`` set to the stage-1 checkpoint).
    """

    CAMERA_INDICES = [3]

    def __init__(
        self,
        config: TransfuserConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
        training_epochs=100,
    ):
        super().__init__()

        self._config = config
        self._lr = lr
        self._training_epochs = training_epochs

        if checkpoint_path is None:
            checkpoint_path = getattr(config, "checkpoint_path", None)
        self._checkpoint_path = checkpoint_path
        self._transfuser_model = V2TransfuserModel(config)
        self.init_from_pretrained()

        if getattr(config, "scorer_only", False):
            self._freeze_except(self._scorer_trainable_prefixes())

    def _scorer_trainable_prefixes(self):
        """Module prefixes left trainable in the scorer-only (stage 2) finetune; everything else is frozen."""
        return [
            '_trajectory_head.plan_anchor_scorer_encoder',
            '_trajectory_head.scorer_decoder',
            '_trajectory_head.scorer_status_proj',
            '_trajectory_head.NC_head',
            '_trajectory_head.DAC_head',
            '_trajectory_head.EP_head',
            '_trajectory_head.C_head',
            '_trajectory_head.TTC_head',
            '_trajectory_head.area_pred_head',
            '_trajectory_head.ttc_time_head',
        ]

    def _freeze_except(self, trainable_prefixes):
        self._transfuser_model.eval()
        for name, param in self._transfuser_model.named_parameters():
            param.requires_grad = any(name.startswith(p) for p in trainable_prefixes)
        for p in trainable_prefixes:
            m = dict(self._transfuser_model.named_modules()).get(p, None)
            if m is not None:
                m.train()

    def init_from_pretrained(self):
        if self._checkpoint_path:
            if torch.cuda.is_available():
                checkpoint = torch.load(self._checkpoint_path, weights_only=True)
            else:
                checkpoint = torch.load(self._checkpoint_path, map_location=torch.device('cpu'), weights_only=True)

            state_dict = checkpoint['state_dict']
            state_dict = {k.replace('agent.', ''): v for k, v in state_dict.items()}

            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

            if missing_keys:
                print(f"Missing keys when loading pretrained weights: {missing_keys}")
            if unexpected_keys:
                assert False, f"Unexpected keys when loading pretrained weights: {unexpected_keys}"
        else:
            print("No checkpoint path provided. Initializing from scratch.")

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, weights_only=True)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(
                self._checkpoint_path, map_location=torch.device("cpu"), weights_only=True
            )["state_dict"]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def compute_trajectory(self, agent_input: AgentInput, multi_return=False) -> Trajectory:
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        features = {k: v.unsqueeze(0) for k, v in features.items()}

        with torch.no_grad():
            predictions = self.forward(features, multi_return=multi_return)
            poses = predictions["trajectory"].squeeze(0).numpy()

        if not multi_return:
            return Trajectory(poses)
        else:
            return poses

    def get_sensor_config(self) -> SensorConfig:
        idx = self.CAMERA_INDICES
        return SensorConfig(
            cam_f0=idx,
            cam_l0=idx,
            cam_l1=idx,
            cam_l2=idx,
            cam_r0=idx,
            cam_r1=idx,
            cam_r2=idx,
            cam_b0=idx,
            lidar_pc=[0, 1, 2, 3],
        )

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [TransfuserTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [TransfuserFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor] = None, multi_return=False) -> Dict[str, torch.Tensor]:
        return self._transfuser_model(features, targets=targets, multi_return=multi_return)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return transfuser_loss(targets, predictions, self._config)

    def get_optimizers(self) -> Dict:
        optimizer = torch.optim.AdamW(
            self._transfuser_model.parameters(),
            lr=self._lr,
            weight_decay=self._config.weight_decay,
        )
        warmup_epochs = max(1, int(0.03 * self._training_epochs))
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self._lr,
            min_lr=1e-6,
            epochs=self._training_epochs,
            warmup_epochs=warmup_epochs,
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_training_callbacks(self) -> List[pl.Callback]:
        return [TransfuserCallback(self._config)]
