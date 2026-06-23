import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

@dataclass
class TransfuserConfig:
    """Global TransFuser config with array reward encoding enabled."""

    trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

    image_architecture: str = "resnet34"
    lidar_architecture: str = "resnet34"
    # Local pretrained backbone weights. Leave unset to download from timm.
    # Set FLOWR2A_BKB_PATH to a local pytorch_model.bin to load offline.
    bkb_path: str = os.environ.get("FLOWR2A_BKB_PATH", "")


    latent: bool = False
    latent_rad_thresh: float = 4 * np.pi / 9

    max_height_lidar: float = 100.0
    pixels_per_meter: float = 4.0
    hist_max_per_pixel: int = 5

    lidar_min_x: float = -32
    lidar_max_x: float = 32
    lidar_min_y: float = -32
    lidar_max_y: float = 32

    lidar_split_height: float = 0.2
    use_ground_plane: bool = True

    # new
    lidar_seq_len: int = 1

    camera_width: int = 1024
    camera_height: int = 256
    lidar_resolution_width = 256
    lidar_resolution_height = 256

    img_vert_anchors: int = 256 // 32
    img_horz_anchors: int = 1024 // 32
    lidar_vert_anchors: int = 256 // 32
    lidar_horz_anchors: int = 256 // 32

    block_exp = 4
    n_layer = 2  # Number of transformer layers used in the vision backbone
    n_head = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    # Mean of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_mean = 0.0
    # Std of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_std = 0.02
    # Initial weight of the layer norms in the gpt.
    gpt_layer_norm_init_weight = 1.0

    perspective_downsample_factor = 1
    transformer_decoder_join = True
    detect_boxes = True
    use_bev_semantic = True
    use_semantic = False
    use_depth = False
    add_features = True

    # Transformer
    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0

    # FlowR2A
    inference_num_timesteps = 20
    transformer_n_layers = 4
    num_traj_sample = 20
    # Trajectory vocabulary for training-time sampling. Not needed for inference
    # (the vocab is restored from the checkpoint). Set FLOWR2A_TRAJ_LIST_PATH to train.
    traj_list_path = os.environ.get("FLOWR2A_TRAJ_LIST_PATH", "")
    status_len = 44 # 11 * 4 frames
    lidar_seq_len = 4
    encoder_config = {
        "condition_dim": 256,
        "noisy_score" : True,
        "noise_scale" : 0.05,
        "ego_progress_key":"safe_ego_progress_normed",
        "reward_to_drop": ['time_to_collision_within_bound', 'drivable_area_compliance'],
        "cond_ttc_time_clip": 1.5,
    }
    inference_pdm_score = 0.95
    decoder_config = {
        "sin_dim": 64,
    }
    n_scorer_layers = 2
    inference_init_step = 10

    # Proposal-centric prediction heads
    area_pred: bool = False

    # TTC time prediction head
    ttc_time_pred: bool = False
    ttc_time_weight: float = 1.0

    # Stage-2 (scorer-only) training: defaults are the stage-1 values.
    # The stage-2 agent yaml overrides these (scorer_only=True, a stage-1 checkpoint_path,
    # and num_scorer_traj_sample=32).
    scorer_only: bool = False
    checkpoint_path: Optional[str] = None
    num_scorer_traj_sample: int = 20  # stage-1 falls back to num_traj_sample (=20)

    # PDM cache directory name
    pdm_cache_dir: str = "train_pdm_cache"

    # detection
    num_bounding_boxes: int = 30

    # loss weights
    trajectory_weight: float = 40.0
    agent_class_weight: float = 10.0
    agent_box_weight: float = 1.0
    bev_semantic_weight: float = 14.0
    transfuser_trajectory_weight: float = 10.0
    score_weight = 10.0

    # BEV mapping
    bev_semantic_classes = {
        1: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]),  # road
        2: ("polygon", [SemanticMapLayer.WALKWAYS]),  # walkways
        3: ("linestring", [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]),  # centerline
        4: (
            "box",
            [
                TrackedObjectType.CZONE_SIGN,
                TrackedObjectType.BARRIER,
                TrackedObjectType.TRAFFIC_CONE,
                TrackedObjectType.GENERIC_OBJECT,
            ],
        ),  # static_objects
        5: ("box", [TrackedObjectType.VEHICLE]),  # vehicles
        6: ("box", [TrackedObjectType.PEDESTRIAN]),  # pedestrians
    }

    bev_pixel_width: int = lidar_resolution_width
    bev_pixel_height: int = lidar_resolution_height // 2
    bev_pixel_size: float = 0.25

    num_bev_classes = 7
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2


    # optimizer
    weight_decay: float = 1e-4

    # Inference parameters
    test_score_min: float = 0.9
    test_score_max: float = 1.0
    test_num_traj_sampling: int = 60
    test_weight_ttc: float = 1.0
    test_weight_ep: float = 1.0
    test_weight_c: float = 1.0

    # Number of trajectories sampled per frame when generating scorer-training proposals.
    inference_num_traj_sampling: int = 60
    # Classifier-free guidance scale used in the denoising loop.
    cfg_scale: float = 5.0

    init_step_min: int = 10
    init_step_max: int = 18

    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_radius(self) -> float:
        values = [self.lidar_min_x, self.lidar_max_x, self.lidar_min_y, self.lidar_max_y]
        return max([abs(value) for value in values])
