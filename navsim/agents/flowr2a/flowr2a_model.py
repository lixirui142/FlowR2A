import os
from typing import Dict, Tuple, Optional, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import OmegaConf
import concurrent.futures as cf
import multiprocessing as mp

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from torch.utils.data import default_collate

from navsim.agents.flowr2a.flowr2a_config import TransfuserConfig
from navsim.agents.flowr2a.flowr2a_backbone import TransfuserBackbone
from navsim.agents.flowr2a.flowr2a_features import BoundingBox2DIndex
from navsim.agents.flowr2a.modules.blocks import (
    gen_sineembed_for_position_1d,
    linear_relu_ln,
    gen_sineembed_for_position,
)
from navsim.agents.transfuser.transfuser_model import TrajectoryHead as TransfuserTrajectoryHead

# Extracted modules
from navsim.agents.flowr2a.pdm_scoring import (
    _pdm_worker,
    _init_pool,
)
from navsim.agents.flowr2a.decoder_modules import (
    CustomTransformerDecoderLayer,
    CustomTransformerDecoder,
)
from navsim.agents.flowr2a.scorer_modules import (
    ScorerTransformerDecoderLayer,
    ScorerTransformerDecoder,
)
from navsim.agents.flowr2a.reward_encoders import RewardEncoderV7


# Default PDM scoring config path
_DEFAULT_PDM_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    '../../planning/script/config/pdm_scoring/default_scoring_parameters.yaml'
)


class V2TransfuserModel(nn.Module):
    """Torch module for Transfuser."""

    def __init__(self, config: TransfuserConfig):
        """
        Initializes TransFuser torch module.
        :param config: global config dataclass of TransFuser.
        """

        super().__init__()

        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]

        self._config = config

        self._backbone = TransfuserBackbone(config)
        self._keyval_embedding = nn.Embedding(config.lidar_vert_anchors * config.lidar_horz_anchors + 1, config.tf_d_model)
        self._bev_downscale = nn.Conv2d(self._backbone.num_features, config.tf_d_model, kernel_size=1)

        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)
        self._status_encoding = nn.Linear(config.status_len, config.tf_d_model)

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                mode="bilinear",
                align_corners=False,
            ),
        )

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._transfuser_trajectory_head = TransfuserTrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._trajectory_head = TrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            n_layers=config.transformer_n_layers,
            config=config,
        )

        # Projects the concatenated cross-BEV feature (keyval BEV @ tf_d_model + upscaled
        # backbone BEV @ bev_features_channels) back down to tf_d_model.
        self.bev_proj = nn.Sequential(
            *linear_relu_ln(config.tf_d_model, 1, 1, config.tf_d_model + config.bev_features_channels),
        )


        

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None, multi_return=False) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        camera_feature: torch.Tensor = features["camera_feature"]
        status_feature: torch.Tensor = features["status_feature"]

        tokens: torch.Tensor = features["token"] if "token" in features else None
        reward_list = features["reward"] if "reward" in features else None
        token_paths = features["token_path"] if "token_path" in features else None
        batch_size = status_feature.shape[0]

        # TransfuserBackbone path (ResNet34)
        camera_feature = camera_feature[-1] if isinstance(camera_feature, list) else camera_feature
        lidar_feature: torch.Tensor = features["lidar_feature"]
        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
        cross_bev_feature = bev_feature_upscale
        bev_spatial_shape = bev_feature_upscale.shape[2:]
        concat_cross_bev_shape = bev_feature.shape[2:]
        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)

        status_encoding = self._status_encoding(status_feature)
        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}

        agents = self._agent_head(agents_query)
        output.update(agents)

        transfuser_trajectory = self._transfuser_trajectory_head(trajectory_query)["trajectory"]
        output["transfuser_trajectory"] = transfuser_trajectory

        concat_cross_bev = keyval[:,:-1].permute(0,2,1).contiguous().view(batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1])
        concat_cross_bev = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode='bilinear', align_corners=False)
        cross_bev_feature = torch.cat([concat_cross_bev, cross_bev_feature], dim=1)
        cross_bev_feature_bnc = self.bev_proj(cross_bev_feature.flatten(-2,-1).permute(0,2,1))
        cross_bev_feature_bchw = cross_bev_feature_bnc.permute(0,2,1).contiguous().view(batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1])

        traj_keyval = keyval

        trajectory_output = self._trajectory_head(
            query_out,
            traj_keyval,
            token=tokens,
            bev_spatial_shape=bev_spatial_shape,
            anchor_trajectory=transfuser_trajectory,
            multi_return=multi_return,
            bev_feature=cross_bev_feature_bchw,
            token_paths=token_paths,
            reward_list=reward_list,
            status_encoding=status_encoding,
        )
        output.update(trajectory_output)

        return output

class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size() + 2), # add vx,vy
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}


class TrajectoryHead(nn.Module):
    """ FlowR2A trajectory head with reward encoder, flow-matching action decoder and scorer."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int, n_layers: int, config: TransfuserConfig):
        """
        Initializes trajectory head.
        :param num_poses: number of (x,y,theta) poses to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        :param n_layers: number of decoder layers
        :param config: global TransfuserConfig
        """
        super(TrajectoryHead, self).__init__()

        # Trajectory / action layout constants, fixed by the architecture: 8 future poses
        # (4s @ 0.5s), the (x, y, sin, cos) normalized action, the (x, y, heading) raw action,
        # and the [-self.norm_range, self.norm_range] normalization range.
        self.horizon = 8
        self.action_dim_out = 4
        self.action_dim_ori = 3
        self.norm_range = 2

        # Trajectory normalization offsets/spans: x,y are mapped to [-norm_range, norm_range]
        # via 2*R*(v + offset)/span - R, i.e. v in [-offset, span - offset] -> [-R, R].
        self.norm_x_offset = 1.2
        self.norm_x_span = 56.9
        self.norm_y_offset = 20
        self.norm_y_span = 46

        self._d_model = d_model
        self._d_ffn = d_ffn
        self._n_layers = n_layers

        self.diffusion_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
        )

        
        # Training trajectory vocabulary, shape [8192, 8, 3].
        traj_list_path = config.traj_list_path
        if traj_list_path and os.path.exists(traj_list_path):
            traj_list = np.load(traj_list_path)
            self.traj_list = nn.Parameter(
                torch.tensor(traj_list[:, 4::5, :], dtype=torch.float32),
                requires_grad=False,
            )
        else:
            # no vocab file -> leave zero;
            print(f"[FlowR2A] traj_list not found at '{traj_list_path}'; "
                  "skipping vocab load.")
            self.traj_list = nn.Parameter(
                torch.zeros([8192,8,3], dtype=torch.float32),
                requires_grad=False,
            )

        # Flow-based action decoder transformer layer
        # Reward + timestep embedding injected via AdaLN -> ada_dim = 2 * d_model
        ada_dim = config.tf_d_model * 2
        dec_dropout = getattr(config, 'dec_dropout', config.tf_dropout)
        diff_decoder_layer = CustomTransformerDecoderLayer(
            d_model=d_model,
            d_ffn=d_ffn,
            num_head=config.tf_num_head,
            dropout=dec_dropout,
            ada_dim=ada_dim,
        )
        # Flow-based action decoder
        decoder_kwargs = getattr(config, "decoder_config", {})
        self.diff_decoder = CustomTransformerDecoder(
            diff_decoder_layer, n_layers, d_model,
            action_dim=self.action_dim_out,
            horizon=self.horizon,
            **decoder_kwargs,
        )

        # Reward encoder
        self.reward_cond_encoder = RewardEncoderV7(**dict(config.encoder_config))
        self.num_rewards = self.reward_cond_encoder.num_rewards

        # Initialize config-driven attributes with defaults
        self._init_config_defaults(config)

        # Whether forward_test runs in evaluation mode or
        # training-validation mode; set via the EVAL env var.
        self.eval_mode = bool(int(os.environ.get("EVAL", "0")))
        print(f"Evaluation Mode: {self.eval_mode}")

        self.init_score_head(config=config)

        # PDM scoring setup (for stage 2 training and validation)
        pdm_config_path = getattr(config, "pdm_scoring_config_path", _DEFAULT_PDM_CONFIG_PATH)
        pdm_cfg = OmegaConf.load(pdm_config_path)
        self.simulator_cfg = pdm_cfg.simulator
        self.scorer_cfg = pdm_cfg.scorer
        
        self._pdm_pool = cf.ProcessPoolExecutor(
            max_workers=4,
            mp_context=mp.get_context("spawn"),
            initializer=_init_pool,
            initargs=(self.simulator_cfg, self.scorer_cfg, self._ttc_bound, self._use_history_comfort, self._driving_direction_exclude_intersection),
        )
        if not self.eval_mode:
            self._pdm_pool_val = cf.ProcessPoolExecutor(
                max_workers=4,
                mp_context=mp.get_context("spawn"),
                initializer=_init_pool,
                initargs=(self.simulator_cfg, self.scorer_cfg, 1.0, False, False),
            )
        self.metric_caches = {}

        # Backward-compat aliases: some scripts still reference the old suffixed names.
        # TODO: remove these once callers migrate to the no-suffix methods.
        self.inference_v4 = self.inference
        self.forward_test_v1 = self.forward_test

    def _init_config_defaults(self, config):
        """Initialize config-driven attributes (with defaults where optional)."""
        # Trajectory sampling / normalization
        self.sampling_alpha = 0.6
        self.num_traj_sample = config.num_traj_sample
        self.num_scorer_traj_sample = getattr(config, "num_scorer_traj_sample", self.num_traj_sample)
        # x,y range assumed same
        self.lidar_max = config.lidar_max_x
        self.lidar_min = config.lidar_min_x

        # Inference / denoising
        self.inference_num_timesteps = config.inference_num_timesteps
        self.inference_pdm_score = getattr(config, "inference_pdm_score", 0.95)
        self.inference_init_step = getattr(config, "inference_init_step", 0)
        self.inference_num_traj_sampling = getattr(config, "inference_num_traj_sampling", 60)
        self.cfg_scale = getattr(config, "cfg_scale", 5.0)
        self.p_partial = 0.4  # CFG partial-drop mode probability
        self.scorer_only = getattr(config, "scorer_only", False)

        self.area_pred = getattr(config, "area_pred", False)
        self.ttc_time_pred = getattr(config, "ttc_time_pred", False)
        self.pdm_cache_dir = getattr(config, "pdm_cache_dir", "train_pdm_cache")
        self.validation_ttc_bound = getattr(config, "validation_ttc_bound", 1.0)

        # PDM scoring / TTC options (consumed by the process-pool workers and ttc transforms)
        self._ttc_bound = getattr(config, "ttc_bound", 2.0)
        self._use_history_comfort = getattr(config, "use_history_comfort", True)
        self._driving_direction_exclude_intersection = getattr(config, "driving_direction_exclude_intersection", True)

        # forward_test_v1 inference params
        self.test_score_min = getattr(config, "test_score_min", 0.9)
        self.test_score_max = getattr(config, "test_score_max", 1.0)
        self.test_num_traj_sampling = getattr(config, "test_num_traj_sampling", 60)
        # Inference scoring weights: NC and DAC are gating (multiplied), others are a weighted sum.
        self.test_weight_ttc = getattr(config, "test_weight_ttc", 1.0)
        self.test_weight_ep = getattr(config, "test_weight_ep", 1.0)
        self.test_weight_c = getattr(config, "test_weight_c", 1.0)

        # inference_v4 parameters (init-step + score are always sampled uniformly)
        self.init_step_min = getattr(config, "init_step_min", 10)
        self.init_step_max = getattr(config, "init_step_max", 18)

        # iPad-style prediction loss weights
        self.pred_area_weight = getattr(config, "pred_area_weight", 2.0)
        self.ttc_time_weight = getattr(config, "ttc_time_weight", 1.0)

    def _apply_ttc_nofix(self, rewards_dict):
        """Overwrite the last 7 timesteps of cached ttc_time with ttc_time_nofix to align
        cached values with online scoring."""
        if "ttc_time" in rewards_dict and "ttc_time_nofix" in rewards_dict:
            rewards_dict["ttc_time"][..., 33:] = rewards_dict["ttc_time_nofix"]
        return rewards_dict

    def _apply_ttc_transforms(self, rewards_dict):
        """Recompute the continuous TTC score from ttc_time and clip ttc_time (stage-2 scorer training).

        ``time_to_collision_within_bound`` is recomputed as the per-trajectory minimum TTC
        normalized by the 2.0s bound.
        """
        if "ttc_time" in rewards_dict:
            min_ttc = rewards_dict["ttc_time"].min(dim=-1)[0]
            rewards_dict["time_to_collision_within_bound"] = (min_ttc / 2.0).clamp(max=1.0)

        return rewards_dict

    def _replace_with_v1_rewards(self, reward_list):
        """Replace reward keys with their v1 variants (stage-2 training)."""
        v1_keys = {
            'no_at_fault_collisions': 'no_at_fault_collisions_v1',
            'drivable_area_compliance': 'drivable_area_compliance_v1',
            'driving_direction_compliance': 'driving_direction_compliance_v1',
            'ego_progress': 'ego_progress_v1',
            'time_to_collision_within_bound': 'time_to_collision_within_bound_v1',
            'history_comfort': 'history_comfort_v1',
        }
        for orig_key, v1_key in v1_keys.items():
            if v1_key in reward_list:
                reward_list[orig_key] = reward_list[v1_key].float()
        return reward_list

    # downsample 40 timesteps to 8 timesteps. [4::5]
    _EGO_AREAS_INDICES = torch.tensor([4, 9, 14, 19, 24, 29, 34, 39])

    def _downsample_ego_areas(self, rewards_dict):
        """Downsample ego_areas from 40 to 8 timesteps if needed."""
        if "ego_areas" not in rewards_dict or rewards_dict["ego_areas"].shape[-2] != 40:
            return rewards_dict
        ea = rewards_dict["ego_areas"]
        idx = self._EGO_AREAS_INDICES.to(ea.device)
        rewards_dict["ego_areas"] = ea.index_select(-2, idx)
        return rewards_dict

    def _sample_timesteps(self, bs, device):
        return torch.rand((bs,), device=device)

    def get_pdm_score_para(self, trajectory, metric_cache_path):
        B = trajectory.shape[0]
        traj_np = trajectory.detach().cpu().numpy()
        futures = [
            self._pdm_pool.submit(
                _pdm_worker,
                (metric_cache_path[b], traj_np[b]),
            )
            for b in range(B)
        ]

        scores_np = np.vstack([f.result()[0] for f in futures])
        sub_scores  = [f.result()[1] for f in futures]

        return torch.from_numpy(scores_np).to(trajectory.device), sub_scores

    def get_pdm_score_para_validation(self, trajectory, metric_cache_path):
        """Get PDM scores using validation pdm scorer."""
        B, G = trajectory.shape[:2]
        traj_np = trajectory.detach().cpu().numpy()
        futures = [
            self._pdm_pool_val.submit(
                _pdm_worker,
                (metric_cache_path[b], traj_np[b]),
            )
            for b in range(B)
        ]
        scores_np = np.vstack([f.result()[0] for f in futures])
        sub_scores = [f.result()[1] for f in futures]
        return torch.from_numpy(scores_np).to(trajectory.device), sub_scores

    def norm_odo(self, odo_info_fut):
        # Norm the trajs to [-2,2]
        x = odo_info_fut[..., 0:1]
        y = odo_info_fut[..., 1:2]
        sin = odo_info_fut[..., 2:3].sin()
        cos = odo_info_fut[..., 2:3].cos()

        x = 2 * self.norm_range * (x + self.norm_x_offset) / self.norm_x_span - self.norm_range
        y = 2 * self.norm_range * (y + self.norm_y_offset) / self.norm_y_span - self.norm_range

        return torch.cat([x, y, sin, cos], dim=-1)

    def denorm_odo(self, odo_info_fut):
        x = odo_info_fut[..., 0:1]
        y = odo_info_fut[..., 1:2]
        sin = odo_info_fut[..., 2:3]
        cos = odo_info_fut[..., 3:4]

        x = (x + self.norm_range) / (2 * self.norm_range) * self.norm_x_span - self.norm_x_offset
        y = (y + self.norm_range) / (2 * self.norm_range) * self.norm_y_span - self.norm_y_offset
        heading = torch.atan2(sin, cos)
        return torch.cat([x, y, heading], dim=-1)

    def forward(self, agents_query, keyval, token = None, bev_spatial_shape = None, anchor_trajectory = None, multi_return = False, bev_feature = None, token_paths = None, reward_list = None, status_encoding = None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        # Params shared by every branch; each branch adds only the extras it consumes.
        kwargs = dict(
            bev_spatial_shape=bev_spatial_shape,
            bev_feature=bev_feature,
            status_encoding=status_encoding,
        )

        if self.training:
            return self.forward_train(agents_query, keyval, reward_list=reward_list, **kwargs)
        elif hasattr(self, "EP_head") and self.EP_head.training:
            return self.forward_train_scorer(agents_query, keyval, token=token, anchor_trajectory=anchor_trajectory, token_paths=token_paths, reward_list=reward_list, **kwargs)
        else:
            return self.forward_test(agents_query, keyval, token=token, anchor_trajectory=anchor_trajectory, multi_return=multi_return, token_paths=token_paths, **kwargs)

    def traj_sampling(self, traj_list, reward_dict, device="cuda", sampling_num=None):
        """
        Args:
            traj_list: Candidate trajectory pool, shape [N, 8, 3].
            reward_dict: Dict of metric tensors, each value shape [B, N]
            device: Compute device
            sampling_num: Number of samples per batch element (K)
        Returns:
            sampled_trajs: [B, sampling_num, 8, 3]
            sampled_rewards: Dict, each value shape [B, sampling_num]
        """

        # 1. Prepare data
        if sampling_num is None:
            sampling_num = self.num_traj_sample

        # Get density values [B, N]
        density = reward_dict["prob_density"].to(device)

        # 2. Compute sampling weights (vectorized)
        epsilon = 1e-8
        inverse_density = 1.0 / (density + epsilon)

        # Apply alpha exponent
        weights = torch.pow(inverse_density, self.sampling_alpha)

        # Normalize along N dimension: [B, N] / [B, 1] -> [B, N]
        weight_sums = weights.sum(dim=1, keepdim=True)
        weights = weights / (weight_sums + 1e-10)

        # 3. Sample indices (vectorized), replacement=True
        # resampled_indices: [B, sampling_num]
        resampled_indices = torch.multinomial(weights, sampling_num, replacement=True)

        # 4. Gather sampled trajectories: [N, 8, 3] indexed by [B, K] -> [B, K, 8, 3]
        sampled_trajs = traj_list[resampled_indices]

        sampled_rewards = {}
        B, K = resampled_indices.shape
        batch_indices = torch.arange(B, device=device).unsqueeze(1)  # [B, 1]
        for key, value in reward_dict.items():
            sampled_rewards[key] = value[batch_indices, resampled_indices]

        return sampled_trajs, sampled_rewards

    def generate_drop_mask(
        self,
        batch_size,
        num_rewards,
        p_uncond=0.10,    # Mode 1: Unconditional (drop all) probability
        indep_prob=0.5,   # Mode 2: per-metric independent drop probability
        device='cpu'
    ):
        """
        Generate drop mask for Classifier-Free Guidance training.

        Modes:
        1. Unconditional (p_uncond): all mask = True
        2. Partial (p_partial): each metric independently set True with indep_prob
        3. Full Conditional (remaining): all mask = False
        """
        p_partial = self.p_partial

        # Random mode selection per sample: [B]
        mode_probs = torch.rand(batch_size, device=device)

        # Initialize mask as False (Full Conditional by default)
        drop_mask = torch.zeros((batch_size, num_rewards), dtype=torch.bool, device=device)

        # --- Mode 1: Unconditional (drop all) ---
        is_uncond = mode_probs < p_uncond
        # Set all metrics to True for unconditional samples
        drop_mask[is_uncond, :] = True

        if p_partial == 0:
            return drop_mask

        # --- Mode 2: Partial / Independent Drop ---
        is_partial = (mode_probs >= p_uncond) & (mode_probs < (p_uncond + p_partial))

        # Only compute if any samples fall into Partial mode
        if is_partial.any():
            # Independent Bernoulli mask [B, N]
            random_drop_matrix = torch.rand((batch_size, num_rewards), device=device) < indep_prob

            # Update only Partial-mode rows
            drop_mask[is_partial, :] = random_drop_matrix[is_partial, :]

        # --- Mode 3: Full Conditional (keep all) ---
        # Implicitly handled: rows not matching Uncond or Partial stay all-False

        return drop_mask

    def create_inference_rewards(
        self,
        batch_size: int,
        device: torch.device,
        rewards_to_keep: Optional[List[str]] = None,
        rewards_to_drop: Optional[List[str]] = None,
        reward_values: Optional[Dict[str, Union[float, torch.Tensor]]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Create reward dictionary and drop mask for inference.

        Args:
            batch_size: Batch size
            device: Device to create tensors on
            rewards_to_keep: List of reward names to keep (use max value).
                            If specified, all others are dropped.
            rewards_to_drop: List of reward names to drop (use null embedding).
                            Ignored if rewards_to_keep is specified.
            reward_values: Optional dict of custom reward values.
                          Default: all rewards = 1.0 (max for scalars), safe defaults for arrays.
                          Can include both scalar and array rewards.
                          - Scalar: float or Tensor[batch_size]
                          - Array: Tensor with appropriate shape (e.g., [batch_size, 40] for ttc_time)

        Returns:
            rewards_dict: Dictionary of reward tensors
            drop_mask: Boolean mask [batch_size, num_rewards] (True = drop)

        Examples:
            # Standard inference: keep NC, DAC, TTC, PDM
            rewards, mask = self.create_inference_rewards(
                batch_size=4,
                device=device,
                reward_values={'pdm_score': 0.95},
                rewards_to_keep=['no_at_fault_collisions', 'drivable_area_compliance',
                               'time_to_collision_within_bound', 'pdm_score']
            )

            # With varying PDM scores
            scores = torch.linspace(0.9, 1.0, steps=20)
            rewards, mask = self.create_inference_rewards(
                batch_size=20,
                device=device,
                reward_values={'pdm_score': scores}
            )

            # With array rewards
            rewards, mask = self.create_inference_rewards(
                batch_size=4,
                device=device,
                reward_values={
                    'pdm_score': 0.95,
                    'ttc_time': torch.full((4, 40), 1.5),  # Safe TTC at all timesteps
                    'ego_areas': torch.zeros((4, 8, 2))    # No violations
                },
                rewards_to_keep=['no_at_fault_collisions', 'ttc_time', 'pdm_score']
            )
        """
        # Get all reward keys from encoder
        reward_specs = self.reward_cond_encoder.reward_specs

        # Initialize rewards dict with default values
        rewards_dict = {}
        for name, r_type, param in reward_specs:
            if r_type == 'array':
                # Default array values (safe/optimal)
                if name == 'ttc_time':
                    default_val = torch.full((batch_size, 40), 1.5, device=device)  # Safe TTC
                elif name == 'ego_areas':
                    num_timesteps = 8
                    default_val = torch.zeros((batch_size, num_timesteps, 2), device=device) # No violations
                else:
                    default_val = torch.ones((batch_size, param), device=device)
            else:
                # Scalar rewards default to 1.0 (max)
                default_val = torch.ones(batch_size, device=device)

            rewards_dict[name] = default_val

        # Override with custom reward values if provided
        if reward_values is not None:
            for name, value in reward_values.items():
                if isinstance(value, (int, float)):
                    rewards_dict[name] = torch.full((batch_size,), value, device=device)
                else:
                    rewards_dict[name] = value.to(device)

        # Create drop mask
        drop_mask = torch.zeros((batch_size, self.num_rewards), dtype=torch.bool, device=device)

        if rewards_to_keep is not None:
            # Keep mode: drop everything except specified rewards
            drop_mask = torch.ones((batch_size, self.num_rewards), dtype=torch.bool, device=device)
            for i, (name, _, _) in enumerate(reward_specs):
                if name in rewards_to_keep:
                    drop_mask[:, i] = False
        elif rewards_to_drop is not None:
            # Drop mode: drop only specified rewards
            for i, (name, _, _) in enumerate(reward_specs):
                if name in rewards_to_drop:
                    drop_mask[:, i] = True

        return rewards_dict, drop_mask

    def init_score_head(self, config):
        """Initialize scoring heads."""
        d_model = self._d_model
        self.plan_anchor_scorer_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1, 2 * 64 * self.horizon),
            nn.Linear(d_model, 512),
        )
        self.scorer_status_proj = nn.Linear(d_model, 512)
        scorer_decoder_layer = ScorerTransformerDecoderLayer(
            num_poses=self.horizon,
            d_model=self._d_model,
            d_ffn=self._d_ffn,
            config=config,
        )
        n_scorer_layers = getattr(config, "n_scorer_layers", 1)
        self.scorer_decoder = ScorerTransformerDecoder(scorer_decoder_layer, n_scorer_layers)
        self.NC_head = nn.Sequential(
            *linear_relu_ln(512, 1, 2),
            nn.Linear(512, 1),
        )
        self.EP_head = nn.Sequential(
            *linear_relu_ln(512, 2, 2),
            nn.Linear(512, 1),
        )
        self.DAC_head = nn.Sequential(
            *linear_relu_ln(512, 1, 2),
            nn.Linear(512, 1),
        )
        self.TTC_head = nn.Sequential(
            *linear_relu_ln(512, 1, 2),
            nn.Linear(512, 1),
        )
        self.C_head = nn.Sequential(
            *linear_relu_ln(512, 1, 2),
            nn.Linear(512, 1),
        )

        if self.area_pred:
            # Output: (B, G, T, C) where T=8, C=2 channels [non_drivable, oncoming]
            num_timesteps = 8
            num_channels = 2
            area_output_size = num_timesteps * num_channels
            self.area_pred_head = nn.Sequential(
                *linear_relu_ln(512, 1, 2),
                nn.Linear(512, area_output_size),
            )

        if self.ttc_time_pred:
            # Output: (B, G, 40) - per-timestep TTC time values
            self.ttc_time_head = nn.Sequential(
                *linear_relu_ln(512, 1, 2),
                nn.Linear(512, 40),  # 40 timesteps
            )

        self.rank_loss = torch.nn.MarginRankingLoss(margin=0.1)
        self.loss_bce = nn.BCEWithLogitsLoss()
        self.loss_bce_without_reduce = nn.BCEWithLogitsLoss(reduction='none')
        self.loss_reg = nn.MSELoss()
        self.sigmoid = nn.Sigmoid()

    def _pairwise_rank_loss(self, pred_scores, gt_scores):
        """Compute pairwise margin ranking loss between predicted and ground-truth scores."""
        B, Gk = pred_scores.shape
        idx_i, idx_j = torch.combinations(
            torch.arange(Gk, device=pred_scores.device), r=2
        ).unbind(-1)

        pred_i, pred_j = pred_scores[:, idx_i], pred_scores[:, idx_j]
        gt_i, gt_j = gt_scores[:, idx_i], gt_scores[:, idx_j]

        target = torch.sign(gt_i - gt_j)
        mask = target != 0
        if mask.any():
            return self.rank_loss(pred_i[mask], pred_j[mask], target[mask])
        return torch.tensor(0., device=pred_scores.device)

    def _score(
        self,
        traj_feature: torch.Tensor,                 # (B, Gk, C)
        sub_rewards_group: Dict[str, torch.Tensor],  # each shape=(B, Gk)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the scoring loss with multi-head sub-metric prediction and pairwise ranking."""
        bs = traj_feature.shape[0]
        device = traj_feature.device
        NC_score  = self.NC_head(traj_feature).squeeze(-1)
        EP_score  = self.EP_head(traj_feature).squeeze(-1)
        DAC_score = self.DAC_head(traj_feature).squeeze(-1)
        TTC_score = self.TTC_head(traj_feature).squeeze(-1)
        C_score   = self.C_head(traj_feature).squeeze(-1)
        gt_nc = sub_rewards_group["no_at_fault_collisions"].to(device)
        gt_nc[gt_nc == 0.5] = 0.0

        loss_nc = self.loss_bce(NC_score, gt_nc)
        gt_ep = sub_rewards_group["ego_progress"].to(device)
        loss_ep = self.loss_bce(EP_score, gt_ep)
        loss_dac = self.loss_bce(DAC_score, sub_rewards_group["drivable_area_compliance"].to(device))
        loss_ttc = self.loss_bce(TTC_score, sub_rewards_group["time_to_collision_within_bound"].to(device))
        loss_c   = self.loss_bce_without_reduce(C_score, sub_rewards_group["history_comfort"].to(device))
        mask = (sub_rewards_group["history_comfort"].to(device) != -1)
        loss_c = (loss_c * mask).sum() / (mask.sum() + 1e-6)

        # Pairwise ranking loss on TTC
        gt_ttc = sub_rewards_group["time_to_collision_within_bound"].to(device)
        loss_rank = self._pairwise_rank_loss(TTC_score, gt_ttc)

        # area prediction loss
        loss_area = 0
        if self.area_pred and "ego_areas" in sub_rewards_group:
            pred_area = self.area_pred_head(traj_feature)  # (B, G, T*C)

            num_timesteps = 8
            # Full mode: (B, G, T, 2)
            pred_area = pred_area.view(bs, -1, num_timesteps, 2)
            gt_areas = sub_rewards_group["ego_areas"].to(device)  # (B, G, T, 2)

            loss_area = self.loss_bce(pred_area, gt_areas.float())

        # TTC time prediction loss
        loss_ttc_time = 0
        if self.ttc_time_pred and "ttc_time" in sub_rewards_group:
            pred_ttc_time = self.ttc_time_head(traj_feature)  # (B, G, 40)
            gt_ttc_time = sub_rewards_group["ttc_time"].to(device)  # (B, G, 40)

            # MSE loss for continuous regression
            loss_ttc_time = F.mse_loss(pred_ttc_time, gt_ttc_time)

        loss = (
            loss_nc + loss_ep + loss_dac + loss_ttc + loss_c +
            2 * loss_rank +
            self.pred_area_weight * loss_area +
            self.ttc_time_weight * loss_ttc_time
        )

        loss_dict = {
            "loss_nc": loss_nc,
            "loss_ep": loss_ep,
            "loss_dac": loss_dac,
            "loss_ttc": loss_ttc,
            "loss_c": loss_c,
            "loss_rank": loss_rank,
            "loss_area": loss_area,
            "loss_ttc_time": loss_ttc_time,
        }
        return loss, loss_dict

    def _get_scorer_inputs(self,
                            diffusion_output: torch.Tensor,   # (B, G_all, 8, 3) -- post denorm
                            bs: int,
                            ego_fut_mode: int):
        """
        Returns:
            traj_points_xy, traj_feature
        """
        # --- Trajectory clamping to the lidar range ---
        traj_points = torch.clamp(diffusion_output, min=self.lidar_min, max=self.lidar_max)  # (B,G,8,3)

        # --- xy + heading positional encoding ---
        traj_points_xy = traj_points[..., :2]
        xy_for_embed = traj_points_xy
        heading_for_embed = traj_points[..., 2]
        traj_pos_embed = gen_sineembed_for_position(
                            xy_for_embed, hidden_dim=64
                        ).flatten(-2)                                # (B,G,8*64)
        traj_heading_embed = gen_sineembed_for_position_1d(
                                heading_for_embed, hidden_dim=32
                            ).flatten(-2)                            # (B,G,8*32)

        traj_pos_embed = torch.cat([traj_pos_embed, traj_heading_embed], dim=-1)
        traj_feature   = self.plan_anchor_scorer_encoder(traj_pos_embed)   # (B,G,C_raw)
        traj_feature   = traj_feature.view(bs, ego_fut_mode, -1)           # (B,G,C)

        return traj_points_xy, traj_feature

    def forward_train(self, agents_query, keyval, bev_spatial_shape = None, bev_feature = None, reward_list = None, status_encoding = None) -> Dict[str, torch.Tensor]:

        # 1. Input preparation
        # The reward encoder uses ttc_time computed with 1.5s bound.
        
        if "ttc_time_b15" in reward_list:
            reward_list["ttc_time"] = reward_list["ttc_time_b15"]

        # Downsample ego_areas from 40 to 8 timesteps
        reward_list = self._downsample_ego_areas(reward_list)

        device = agents_query.device

        # sampling training trajectories
        resampled_trajs, resampled_rewards_ori = self.traj_sampling(self.traj_list, reward_list, device = device)
        resampled_rewards = {k: v.view(-1).to(device) for k, v in resampled_rewards_ori.items()}

        target_traj = resampled_trajs.view(-1, self.horizon, self.action_dim_ori)
        

        # 2. Reward encoding
        # random reward dropping in training
        drop_mask = self.generate_drop_mask(agents_query.shape[0] * self.num_traj_sample, self.num_rewards)
        reward_cond_emb = self.reward_cond_encoder(resampled_rewards, drop_mask)


        # 3. Flow-based action decoder

        # repeat each frame by sampled trajectory number
        agents_query = agents_query.repeat_interleave(self.num_traj_sample, dim = 0)
        keyval = keyval.repeat_interleave(self.num_traj_sample, dim = 0)
        
        bs = keyval.shape[0]

        # trajectory normalization
        x_0 = self.norm_odo(target_traj)

        # noise and timesteps sampling
        timesteps = self._sample_timesteps(bs, device)
        t = timesteps.view(-1, 1, 1)
        noise = torch.randn(x_0.shape, device=device)

        # add noise (forward process)
        x_t = t * x_0 + (1 - t) * noise
        v_target = (x_0 - x_t) / torch.clip(1 - t, min = 0.05)

        # action decoder
        # input: noisy trajectory, scene features, timesteps, reward embedding.
        # output: predicted clean trajectory
        x_pred = self.diff_decoder(x_t, keyval, agents_query, timesteps, reward_cond_emb)

        # compute velocity v from predicted x.
        v_pred = (x_pred - x_t) / torch.clip(1 - t, min = 0.05)

        # compute velocity matching loss
        trajectory_loss = F.mse_loss(v_pred, v_target)

        ret_dict = {"trajectory": x_pred,"trajectory_loss":trajectory_loss}

        # 4. Scorer
        # sampling training trajectories
        resampled_trajs, resampled_rewards_ori = self.traj_sampling(self.traj_list, reward_list, device = device, sampling_num = self.num_scorer_traj_sample)
        sub_rewards_group = {k: v.to(device) for k, v in resampled_rewards_ori.items()}
        
        diffusion_output = resampled_trajs
        batch_size = diffusion_output.shape[0]
        # trajectory input embedding.
        traj_points_xy, traj_feature = self._get_scorer_inputs(diffusion_output, batch_size, diffusion_output.shape[1]) # traj_feature: [B, G, C]


        if status_encoding is not None:
            status_proj = self.scorer_status_proj(status_encoding)  # (B, C)
            traj_feature = traj_feature + status_proj[:, None]
        # back to original batch size for scoring
        ego_agents_query = agents_query[::self.num_traj_sample]
        scorer_keyval = keyval[::self.num_traj_sample]
        ego_query, agents_query = ego_agents_query[:,:1], ego_agents_query[:,1:]
        # scoring
        traj_feature_list = self.scorer_decoder(traj_feature, traj_points_xy, bev_feature, bev_spatial_shape, agents_query, ego_query, keyval=scorer_keyval)
        traj_feature = traj_feature_list[-1]
        loss, sub_loss_dict = self._score(traj_feature, sub_rewards_group)

        ret_dict.update({"score_loss":loss,"score_sub_loss_dict": sub_loss_dict})
        return ret_dict

    def forward_train_scorer(self, agents_query, keyval, token = None, bev_spatial_shape = None, bev_feature = None, anchor_trajectory = None, token_paths = None, reward_list = None, status_encoding = None) -> Dict[str, torch.Tensor]:

        # Default reward caches (no v1 suffix) are borrowed from GTRS, which we find have subtle differences against rewards computed by our pdm_scorer_train (with v1 suffix).
        # Stage-2 training uses our rewards variants to match online reward labeling using pdm_scorer_train.
        reward_list = self._replace_with_v1_rewards(reward_list)
        reward_list = self._apply_ttc_nofix(reward_list)

        with torch.no_grad():
            inference_kwargs = dict(
                add_gt=True,
                cfg=True,
                num_traj_sampling=self.inference_num_traj_sampling,
                score_min=0.9,
                score_max=1.0,
            )
            diffusion_output = self.inference(agents_query, keyval, anchor_trajectory, **inference_kwargs)

        batch_size = diffusion_output.shape[0]

        # Construct metric-cache paths online.
        metric_cache = self._build_metric_cache_paths(token_paths, token)

        _, sub_rewards_group = self.get_pdm_score_para(diffusion_output, metric_cache)
        sub_rewards_group = default_collate(sub_rewards_group)
        device = agents_query.device

        # Append vocabulary trajectories (and their rewards) to the scored set
        vocab, vocab_sub_rewards_group = self.traj_sampling(self.traj_list, reward_list, device = device, sampling_num = self.num_scorer_traj_sample)
        sub_rewards_group = {k: torch.cat([v.to(device), vocab_sub_rewards_group[k]], dim = 1) for k, v in sub_rewards_group.items()}
        diffusion_output = torch.cat((diffusion_output, vocab), dim=1)  # (B,G_all,8,3)

        sub_rewards_group = self._downsample_ego_areas(sub_rewards_group)
        # compute the continuous ttc value
        sub_rewards_group = self._apply_ttc_transforms(sub_rewards_group)

        traj_points_xy, traj_feature = self._get_scorer_inputs(diffusion_output, batch_size, diffusion_output.shape[1])
        if status_encoding is not None:
            status_proj = self.scorer_status_proj(status_encoding)  # (B, 512)
            traj_feature = traj_feature + status_proj[:, None]
        ego_agents_query = agents_query
        ego_query, agents_query = ego_agents_query[:,:1], ego_agents_query[:,1:]
        # scorer
        traj_feature_list = self.scorer_decoder(traj_feature, traj_points_xy, bev_feature, bev_spatial_shape, agents_query, ego_query, keyval=keyval)
        traj_feature = traj_feature_list[-1]
        loss, sub_loss_dict = self._score(traj_feature, sub_rewards_group)
        ret_dict = dict()
        ret_dict.update({"score_loss":loss,"score_sub_loss_dict": sub_loss_dict})
        return ret_dict

    def _build_metric_cache_paths(self, token_paths, token):
        """Build per-sample metric_cache.pkl paths from the feature-cache token paths.

        token_path format:   .../training_cache_xxx/scene_dir/token/...
        metric_cache format: .../pdm_cache_dir/scene_dir/unknown/token/metric_cache.pkl
        """
        navsim_root = os.getenv("NAVSIM_EXP_ROOT", "")
        metric_cache = []
        for batch_idx, token_path in enumerate(token_paths):
            token_path_parts = str(token_path).split('/')
            tk = token[batch_idx]
            try:
                token_idx = token_path_parts.index(tk)
                scene_dir = token_path_parts[token_idx - 1]
                pdm_token_path = os.path.join(navsim_root, self.pdm_cache_dir, scene_dir, 'unknown', tk, 'metric_cache.pkl')
            except (ValueError, IndexError):
                pdm_token_path = None
            metric_cache.append(pdm_token_path)
        return metric_cache

    def _index_reward_emb(self, reward_emb, indices):
        """Index the reward conditioning embedding for the active trajectories."""
        return reward_emb[indices]

    def _merge_reward_emb(self, reward_emb_neg, reward_emb_pos):
        """Stack negative (unconditional) and positive reward embeddings for CFG."""
        return torch.cat([reward_emb_neg, reward_emb_pos], dim=0)

    def inference(self, agents_query, keyval, anchor_trajectory=None, add_gt=True, cfg=True, num_traj_sampling=None, score_min=0.9, score_max=1.0):
        """Inference: gradual trajectory injection with per-trajectory init_step and score sampling."""
        num_traj_sampling = self.inference_num_traj_sampling if num_traj_sampling is None else num_traj_sampling
        original_bs = agents_query.shape[0]
        device = agents_query.device
        node_num = self.horizon

        # Sample per-trajectory init_steps (always uniform)
        init_steps_sampled = torch.randint(self.init_step_min, self.init_step_max + 1, (original_bs * num_traj_sampling,), device=device)

        # Sample per-trajectory scores (always uniform)
        pdm_scores = torch.rand(original_bs * num_traj_sampling, device=device) * (score_max - score_min) + score_min

        # Repeat context by num_traj_sampling
        agents_query = agents_query.repeat_interleave(num_traj_sampling, dim=0)
        keyval = keyval.repeat_interleave(num_traj_sampling, dim=0)
        anchor_trajectory = anchor_trajectory.repeat_interleave(num_traj_sampling, dim=0)
        bs = agents_query.shape[0]

        # Pre-compute noise and normalized x_0 for all trajectories
        noise = torch.randn([bs, node_num, self.action_dim_out], device=device)
        x_0 = self.norm_odo(anchor_trajectory)

        # Setup scheduler
        self.diffusion_scheduler.set_timesteps(self.inference_num_timesteps, device)

        # Create reward embeddings for all trajectories (score target: pdm_score)
        rewards_to_keep = ['no_at_fault_collisions', 'drivable_area_compliance',
                           'time_to_collision_within_bound', 'history_comfort',
                           'pdm_score', 'ttc_time', 'ego_areas']

        max_rewards, drop_mask = self.create_inference_rewards(
            batch_size=bs, device=device,
            reward_values={'pdm_score': pdm_scores},
            rewards_to_keep=rewards_to_keep
        )
        reward_cond_emb = self.reward_cond_encoder(max_rewards, drop_mask, eval=True)

        neg_rewards, drop_mask_neg = self.create_inference_rewards(
            batch_size=bs, device=device,
            rewards_to_drop=list(self.reward_cond_encoder.reward_keys)
        )
        reward_cond_emb_neg = self.reward_cond_encoder(neg_rewards, drop_mask_neg, eval=True)

        cfg_scale = self.cfg_scale

        # Gradual injection denoising loop
        x_t_active = None
        active_indices = []
        roll_timesteps = self.diffusion_scheduler.timesteps

        for i, k in enumerate(roll_timesteps):
            # Find trajectories to inject at this step
            new_indices = (init_steps_sampled == i).nonzero(as_tuple=True)[0]

            if len(new_indices) > 0:
                # Inject new trajectories
                # scheduler integer timestep k: 1000 to 0 -> flow matching timestep t: 0 to 1.
                t_init = 1 - (k / 1000)
                x_t_new = t_init * x_0[new_indices] + (1 - t_init) * noise[new_indices]

                if x_t_active is None:
                    x_t_active = x_t_new
                    active_indices = new_indices.tolist()
                else:
                    x_t_active = torch.cat([x_t_active, x_t_new], dim=0)
                    active_indices.extend(new_indices.tolist())

            if x_t_active is None:
                continue

            # Index context and rewards for active trajectories
            active_idx_tensor = torch.tensor(active_indices, device=device)
            agents_query_active = agents_query[active_idx_tensor]
            keyval_active = keyval[active_idx_tensor]
            reward_cond_emb_active = self._index_reward_emb(reward_cond_emb, active_idx_tensor)
            reward_cond_emb_neg_active = self._index_reward_emb(reward_cond_emb_neg, active_idx_tensor)

            # scheduler integer timestep k: 1000 to 0 -> flow matching timestep t: 0 to 1.
            ts = 1 - (k / 1000)

            if cfg:
                x_t_doubled = torch.cat([x_t_active, x_t_active], dim=0)
                agents_query_doubled = torch.cat([agents_query_active, agents_query_active], dim=0)
                keyval_doubled = torch.cat([keyval_active, keyval_active], dim=0)
                reward_cond_emb_merged = self._merge_reward_emb(reward_cond_emb_neg_active, reward_cond_emb_active)
            else:
                x_t_doubled = x_t_active
                agents_query_doubled = agents_query_active
                keyval_doubled = keyval_active
                reward_cond_emb_merged = reward_cond_emb_active

            x_pred = self.diff_decoder(x_t_doubled, keyval_doubled, agents_query_doubled, ts, reward_cond_emb_merged)
            v_pred = (x_pred - x_t_doubled) / (1 - ts)

            if cfg:
                v_pred_neg, v_pred_pos = v_pred.chunk(2, dim=0)
                v_pred = v_pred_neg + cfg_scale * (v_pred_pos - v_pred_neg)

            # ODE step
            timesteps = 1 - self.diffusion_scheduler.sigmas
            timestep = timesteps[i]
            timestep_next = timesteps[i + 1]
            x_t_active = x_t_active + (timestep_next - timestep) * v_pred

        if cfg:
            x_pred = x_pred.chunk(2, dim=0)[1].to(x_t_active)
        # Reshape output
        output = torch.zeros([bs, node_num, self.action_dim_out], device=device)

        # Assign last step output to x_pred. Identical to disable CFG in the last step.
        # Almost no influence on performance. Can be commented out.
        if x_t_active is not None:
            active_idx_tensor = torch.tensor(active_indices, device=device)
            output[active_idx_tensor] = x_pred

        diffusion_output = self.denorm_odo(output).view(original_bs, num_traj_sampling, node_num, self.action_dim_ori)

        if add_gt:
            anchor_trajectory = anchor_trajectory.view(original_bs, num_traj_sampling, node_num, self.action_dim_ori)[:, :1, ...]
            diffusion_output = torch.cat([diffusion_output, anchor_trajectory], dim=1)

        return diffusion_output

    def _compute_final_score(self, traj_feature):
        """Score trajectories from scorer features.

        Returns:
            final_score: (B, G) combined score used to rank trajectories.
            subscore: dict of per-metric sigmoid scores (NC, DAC, TTC, EP, C).
        """
        NC_score = self.NC_head(traj_feature).squeeze(-1)             # (B,G)
        EP_score = self.EP_head(traj_feature).squeeze(-1)             # (B,G)
        DAC_score = self.DAC_head(traj_feature).squeeze(-1)             # (B,G)
        TTC_score = self.TTC_head(traj_feature).squeeze(-1)             # (B,G)
        C_score = self.C_head(traj_feature).squeeze(-1)             # (B,G)
        denorm = self.test_weight_ttc + self.test_weight_ep + self.test_weight_c
        final_score = self.sigmoid(NC_score)*self.sigmoid(DAC_score)*(self.test_weight_ttc*self.sigmoid(TTC_score)+self.test_weight_ep*self.sigmoid(EP_score)+self.test_weight_c*self.sigmoid(C_score))/denorm
        subscore = {"NC": self.sigmoid(NC_score), "DAC": self.sigmoid(DAC_score), "TTC": self.sigmoid(TTC_score), "EP": self.sigmoid(EP_score), "C": self.sigmoid(C_score)}
        return final_score, subscore

    def forward_test(self, agents_query, keyval, bev_spatial_shape = None, anchor_trajectory = None, multi_return=False, bev_feature = None, token_paths = None, token = None, status_encoding = None) -> Dict[str, torch.Tensor]:
        score_min = self.test_score_min
        score_max = self.test_score_max
        device = agents_query.device

        diffusion_output = self.inference(agents_query, keyval, anchor_trajectory, add_gt = True, cfg = True, num_traj_sampling = self.test_num_traj_sampling, score_min = score_min, score_max = score_max)
        batch_size = diffusion_output.shape[0]

        traj_points_xy, traj_feature = self._get_scorer_inputs(diffusion_output, batch_size, diffusion_output.shape[1])
        if status_encoding is not None:
            status_proj = self.scorer_status_proj(status_encoding)  # (B, 512)
            traj_feature = traj_feature + status_proj[:, None]
        ego_agents_query = agents_query
        ego_query, agents_query_split = ego_agents_query[:,:1], ego_agents_query[:,1:]

        traj_feature_list = self.scorer_decoder(traj_feature, traj_points_xy, bev_feature, bev_spatial_shape, agents_query_split, ego_query, keyval=keyval)

        traj_feature = traj_feature_list[-1]
        final_score, subscore = self._compute_final_score(traj_feature)

        best_idx = torch.argmax(final_score, dim=-1)      # (B,)

        traj_to_score = diffusion_output[
            torch.arange(batch_size, device=device), best_idx
        ].unsqueeze(1)  # (B,1,8,3)

        # Normal inference
        if self.eval_mode:
            if multi_return:
                meta_infos = {
                    "score_min": self.test_score_min,
                    "score_max": self.test_score_max,
                    "num_traj_sampling": self.test_num_traj_sampling,
                    "weights": {"ttc": self.test_weight_ttc, "ep": self.test_weight_ep, "c": self.test_weight_c, "denom": 3.0},
                    "init_step_min": self.init_step_min,
                    "init_step_max": self.init_step_max,
                }
                return {"trajectory": traj_to_score[:,-1], "trajs": diffusion_output, "scores": final_score, "subscore": subscore, "best_idx": best_idx, "meta_infos": meta_infos}
            else:
                return {"trajectory": traj_to_score[:,-1]}

        # Validation during training — add PDM scoring
        result = {"trajectory": traj_to_score[:, -1]}

        # [NOTE] Uncomment the following code if you want to enable PDM score logging for validation. 

        # if token_paths is not None and token is not None:
        #     try:
        #         # Construct metric-cache paths online (the DataLoader does not pre-load them).
        #         metric_cache = self._build_metric_cache_paths(token_paths, token)

        #         # 2 trajs: selected, anchor
        #         trajs_to_eval = torch.cat([traj_to_score, diffusion_output[:, [-1], ...]], dim=1)  # (B,2,8,3)
        #         pdm_scores, _, sub_scores, _ = self.get_pdm_score_para_validation(trajs_to_eval, metric_cache)

        #         traj_names = ["selected", "anchor"]
        #         metric_keys = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
        #                        "time_to_collision_within_bound", "history_comfort", "driving_direction_compliance"]
        #         for ti, tname in enumerate(traj_names):
        #             result[f"val_pdm_score_{tname}"] = pdm_scores[:, ti].mean().to(device)
        #             for mkey in metric_keys:
        #                 short_key = {"no_at_fault_collisions": "NC", "drivable_area_compliance": "DAC",
        #                              "ego_progress": "EP", "time_to_collision_within_bound": "TTC",
        #                              "history_comfort": "C", "driving_direction_compliance": "DDC"}[mkey]
        #                 vals = [sub_scores[b][mkey][ti] for b in range(batch_size)]
        #                 result[f"val_pdm_{short_key}_{tname}"] = torch.tensor(vals).float().mean().to(device)
        #     except Exception:
        #         pass  # Don't break training if PDM cache is missing

        return result