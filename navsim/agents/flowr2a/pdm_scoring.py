"""
PDM (Planning-aware Driving Metric) scoring utilities.

Functions for computing pairwise scores between GT and candidate trajectories
using a pre-trained PDMScorer, plus process pool workers for parallel scoring.
"""
import os
import pickle
import lzma
from pathlib import Path
import numpy as np

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    WeightedMetricIndex as WIdx,
    MultiMetricIndex,
    WeightedMetricIndex,
)
from navsim.evaluate.pdm_score import pdm_score_para
from hydra.utils import instantiate

# Process pool globals (set by _init_pool)
SIMULATOR = None
SCORER = None


def _downsample_to_model_resolution(key_agent_corners, key_agent_labels, ego_areas, num_model_poses=8):
    """
    Downsample from 40 simulator timesteps to 8 model timesteps.

    The vocabulary has 40 timesteps at 0.1s intervals: [0.1s, 0.2s, ..., 4.0s]
    Model uses 8 timesteps at 0.5s intervals: [0.5s, 1.0s, 1.5s, 2.0s, 2.5s, 3.0s, 3.5s, 4.0s]
    Correct indices: [4, 9, 14, 19, 24, 29, 34, 39]

    Args:
        key_agent_corners: (..., 40, 2, 4, 2) - timestep at axis -4
        key_agent_labels: (..., 40, 2) - timestep at axis -2
        ego_areas: (..., 40, 3) - timestep at axis -2
        num_model_poses: 8

    Returns:
        Downsampled versions at model resolution
    """
    # Correct indices for [0.5s, 1.0s, 1.5s, 2.0s, 2.5s, 3.0s, 3.5s, 4.0s]
    indices = np.array([4, 9, 14, 19, 24, 29, 34, 39])[:num_model_poses]

    return (
        np.take(key_agent_corners, indices, axis=-4),  # (..., 8, 2, 4, 2)
        np.take(key_agent_labels, indices, axis=-2),   # (..., 8, 2)
        np.take(ego_areas, indices, axis=-2)           # (..., 8, 2)
    )


def _pairwise_subscores(scorer, extract_gt_ep=False):
    """
    Extract 7 sub-metrics and the final score from a scored PDMScorer.

    The scorer must have already called score_proposals(). Returns a dict
    of arrays, each shape=(G,), aligned by proposal index.

    If extract_gt_ep=True, additionally compute 'ego_progress_gt' by normalizing
    each candidate's raw progress against the human GT trajectory (last index)
    instead of the PDM-simulated trajectory (index 0).
    """
    mm = scorer._multi_metrics                # (3, N)
    wm = scorer._weighted_metrics.copy()      # must copy to avoid mutation
    prod = mm.prod(axis=0)                    # (N,)

    wcoef = scorer._config.weighted_metrics_array
    thresh = scorer._config.progress_distance_threshold
    prog_raw = scorer._progress_raw           # (N,)

    # Normalize progress (consistent with _pairwise_scores)
    raw_prog = prog_raw * prod
    raw_prog_gt = raw_prog[0]
    max_pair = np.maximum(raw_prog_gt, raw_prog[1:])
    norm_prog = np.where(
        max_pair > thresh,
        raw_prog[1:] / (max_pair + 1e-6),
        np.where(prod[1:] == 0.0, 0.0, 1.0),
    ).astype(np.float64)
    wm[WeightedMetricIndex.PROGRESS, 1:] = norm_prog

    # Weighted metrics
    wscore = (wm * wcoef[:, None]).sum(axis=0) / wcoef.sum()

    result = {
        "no_at_fault_collisions": mm[MultiMetricIndex.NO_COLLISION, 1:].copy(),
        "drivable_area_compliance": mm[MultiMetricIndex.DRIVABLE_AREA, 1:].copy(),
        "ego_progress": wm[WeightedMetricIndex.PROGRESS, 1:].copy(),
        "time_to_collision_within_bound": wm[WeightedMetricIndex.TTC, 1:].copy(),
        "history_comfort": wm[WeightedMetricIndex.COMFORTABLE, 1:].copy(),
        "driving_direction_compliance": wm[WeightedMetricIndex.DRIVING_DIRECTION, 1:].copy(),
        "pdm_score": prod[1:] * wscore[1:],
    }

    if extract_gt_ep:
        # Normalize progress against human GT (last index) instead of PDM (index 0)
        raw_prog_human_gt = raw_prog[-1]
        max_pair_gt = np.maximum(raw_prog_human_gt, raw_prog[1:])
        norm_prog_gt = np.where(
            max_pair_gt > thresh,
            raw_prog[1:] / (max_pair_gt + 1e-6),
            np.where(prod[1:] == 0.0, 0.0, 1.0),
        ).astype(np.float64)
        result["ego_progress_gt"] = norm_prog_gt

    return result


def _pairwise_subscores_full(scorer):
    """
    Extract the full set of sub-metrics + ego areas + per-step TTC time from a scored scorer.

    Builds on `_pairwise_subscores` (the 7 base PDM sub-metrics) and additionally returns
    `ego_areas` (per-timestep non-drivable / oncoming flags) and `ttc_time` (per-step
    continuous time-to-collision). The scorer must have already called score_proposals().
    All arrays skip the GT trajectory (index 0).
    """
    # First get the standard sub-metrics
    result = _pairwise_subscores(scorer)

    # ego_areas: original EgoAreaIndex has 3 dims (MULTIPLE_LANES=0, NON_DRIVABLE_AREA=1,
    # ONCOMING_TRAFFIC=2). [:,1:,1:] skips timestep 0 and MULTIPLE_LANES → 2 remaining dims.
    ego_areas = scorer._ego_areas[:, 1:, 1:]  # (N, num_poses, 2)

    result["ego_areas"] = ego_areas[1:]

    # Per-timestep continuous TTC: skip GT (index 0) and initial timestep (index 0)
    result["ttc_time"] = scorer._per_step_ttc[1:, 1:]  # (N-1, 40), float32

    return result


def extract_ipad_gt_from_metric_cache(metric_cache, trajectories, compact_ego_areas=False):
    """
    Extract collision agent corners and ego areas from metric cache for Stage 1 training.
    Uses cached observation data without running full PDM simulation.

    Args:
        metric_cache: MetricCache object loaded from pickle
        trajectories: (G, 8, 3) trajectory proposals in ego frame
        compact_ego_areas: If True, return ego_areas as (G, 8) instead of (G, 8, 2)

    Returns:
        key_agent_corners: (G, 8, 2, 4, 2) - collision agent bounding boxes
        key_agent_labels: (G, 8, 2) - collision agent validity masks
        ego_areas: (G, 8, 2) or (G, 8) - ego area flags
    """
    raise RuntimeError(
        "extract_ipad_gt_from_metric_cache() was called but should not be! "
        "All rewards should be pre-cached. Check your reward cache setup."
    )
    from shapely.geometry import Polygon, Point
    from nuplan.common.maps.abstract_map import SemanticMapLayer

    G = len(trajectories)
    key_agent_corners = np.zeros([G, 8, 2, 4, 2], dtype=np.float32)
    key_agent_labels = np.zeros([G, 8, 2], dtype=bool)
    ego_areas = np.zeros([G, 8, 2], dtype=bool)  # Only [non_drivable, oncoming]

    # Get data from cache
    initial_ego = metric_cache.ego_state
    observation = metric_cache.observation
    drivable_map = metric_cache.drivable_area_map

    # Transform trajectories to global frame
    theta = initial_ego.rear_axle.heading
    origin_x, origin_y = initial_ego.rear_axle.x, initial_ego.rear_axle.y
    c, s = np.cos(theta), np.sin(theta)
    rot_mat = np.array([[c, -s], [s, c]])

    trajs_global = trajectories.copy()
    trajs_global[..., :2] = trajs_global[..., :2] @ rot_mat.T
    trajs_global[..., 0] += origin_x
    trajs_global[..., 1] += origin_y
    trajs_global[..., 2] += theta

    # Convert to ego polygons
    vehicle_params = initial_ego.car_footprint
    ego_polygons = _trajectory_to_polygons(trajs_global, vehicle_params)

    # Process each trajectory — find first collision timestep, then fill all prior timesteps
    # (matching iPad's filling strategy: all timesteps from 1 to collision_time)
    for g_idx in range(G):
        # Track first collision timestep per collision type
        first_at_fault_t = None
        first_at_fault_token = None
        first_ttc_t = None
        first_ttc_token = None

        for t_idx in range(8):
            cache_t_idx = t_idx * 5  # 0.5s model -> 0.1s cache (5 steps per model step)

            if cache_t_idx >= len(observation._occupancy_maps):
                continue

            ego_polygon = Polygon(ego_polygons[g_idx, t_idx])
            occupancy = observation._occupancy_maps[cache_t_idx]

            # At-fault collision detection
            if first_at_fault_t is None:
                intersecting = occupancy.intersects(ego_polygon)
                if len(intersecting) > 0:
                    first_at_fault_t = t_idx
                    first_at_fault_token = intersecting[0]

            # TTC collision (1s ahead = 10 steps ahead in cache)
            if first_ttc_t is None:
                ttc_idx = min(cache_t_idx + 10, len(observation._occupancy_maps) - 1)
                occupancy_ttc = observation._occupancy_maps[ttc_idx]
                intersecting_ttc = occupancy_ttc.intersects(ego_polygon)
                if len(intersecting_ttc) > 0:
                    first_ttc_t = t_idx
                    first_ttc_token = intersecting_ttc[0]

            # Ego areas - simplified version (only non_drivable and oncoming)
            ego_center = Point(trajs_global[g_idx, t_idx, 0], trajs_global[g_idx, t_idx, 1])
            try:
                on_drivable = drivable_map.is_in_layer(ego_center, SemanticMapLayer.ROADBLOCK) or \
                              drivable_map.is_in_layer(ego_center, SemanticMapLayer.INTERSECTION)
                ego_areas[g_idx, t_idx, 0] = not on_drivable  # non_drivable flag
            except:
                ego_areas[g_idx, t_idx, 0] = False
            ego_areas[g_idx, t_idx, 1] = False  # oncoming (simplified)

        # Fill all timesteps up to collision (matching iPad's filling strategy)
        if first_at_fault_token is not None:
            for t_idx in range(first_at_fault_t + 1):
                cache_t_idx = t_idx * 5
                if cache_t_idx >= len(observation._occupancy_maps):
                    continue
                occupancy = observation._occupancy_maps[cache_t_idx]
                if first_at_fault_token in occupancy.tokens:
                    agent_poly = occupancy[first_at_fault_token]
                    corners = np.array(agent_poly.exterior.coords)[:4]
                    corners[:, 0] -= origin_x
                    corners[:, 1] -= origin_y
                    corners = corners @ rot_mat.T
                    key_agent_labels[g_idx, t_idx, 0] = True
                    key_agent_corners[g_idx, t_idx, 0] = corners

        if first_ttc_token is not None:
            for t_idx in range(first_ttc_t + 1):
                cache_t_idx = t_idx * 5
                ttc_idx = min(cache_t_idx + 10, len(observation._occupancy_maps) - 1)
                occupancy_ttc = observation._occupancy_maps[ttc_idx]
                if first_ttc_token in occupancy_ttc.tokens:
                    agent_poly_ttc = occupancy_ttc[first_ttc_token]
                    corners_ttc = np.array(agent_poly_ttc.exterior.coords)[:4]
                    corners_ttc[:, 0] -= origin_x
                    corners_ttc[:, 1] -= origin_y
                    corners_ttc = corners_ttc @ rot_mat.T
                    key_agent_labels[g_idx, t_idx, 1] = True
                    key_agent_corners[g_idx, t_idx, 1] = corners_ttc

    # Compact ego_areas: combine violation types via OR
    if compact_ego_areas:
        ego_areas = np.any(ego_areas, axis=-1, keepdims=True).astype(np.float32)  # (G, 8)

    return key_agent_corners, key_agent_labels, ego_areas


def _trajectory_to_polygons(trajectories, vehicle_params):
    """Convert trajectories to vehicle bounding box polygons."""
    G, T = trajectories.shape[:2]
    polygons = np.zeros([G, T, 4, 2], dtype=np.float32)

    half_length = vehicle_params.length / 2
    half_width = vehicle_params.width / 2

    for g in range(G):
        for t in range(T):
            x, y, heading = trajectories[g, t]
            c, s = np.cos(heading), np.sin(heading)

            corners_local = np.array([
                [half_length, half_width],
                [-half_length, half_width],
                [-half_length, -half_width],
                [half_length, -half_width],
            ], dtype=np.float32)

            rot_mat = np.array([[c, -s], [s, c]], dtype=np.float32)
            corners_global = corners_local @ rot_mat.T
            corners_global[:, 0] += x
            corners_global[:, 1] += y

            polygons[g, t] = corners_global

    return polygons


def _pairwise_scores(scorer) -> np.ndarray:
    """
    Recompute "GT (index 0) vs each candidate" scores from cached scorer state.

    Returns shape=(N-1,) float32.
    """
    mm = scorer._multi_metrics              # (M_mul, N)
    wm = scorer._weighted_metrics.copy()    # (M_wgt, N) -- copy to modify progress
    prog_raw = scorer._progress_raw         # (N,)
    weight_coef = scorer._config.weighted_metrics_array  # (M_wgt,)

    N = mm.shape[1]                         # proposals = 1(GT) + G
    assert N >= 2, "Need at least GT + 1 proposal"

    # Multiplicative metric product
    multi_prod = mm.prod(axis=0)            # (N,)

    # Re-normalize progress: each candidate vs GT only
    raw_prog = prog_raw * multi_prod        # (N,)
    raw_prog_gt = raw_prog[0]
    max_pair = np.maximum(raw_prog_gt, raw_prog[1:])          # (G,)
    thresh = scorer._config.progress_distance_threshold

    # If max_pair > thresh: normalize proportionally; else check collision
    norm_prog = np.where(
        max_pair > thresh,
        raw_prog[1:] / (max_pair + 1e-6),
        np.where(multi_prod[1:] == 0.0, 0.0, 1.0),
    ).astype(np.float64)                                       # (G,)

    wm[WIdx.PROGRESS, 1:] = norm_prog

    # Weighted metric scores (same formula as _aggregate_scores)
    weighted_scores = (wm[:, 1:] * weight_coef[:, None]).sum(axis=0)
    weighted_scores /= weight_coef.sum()                       # (G,)

    # Final score = multiplicative * weighted
    final_scores = multi_prod[1:] * weighted_scores            # (G,)
    return final_scores.astype(np.float32)


def _load_metric_cache(cache_path):
    """Load MetricCache from disk, preferring fast plain pickle over lzma.

    Looks for metric_cache_fast.pkl (plain pickle) first; falls back to
    metric_cache.pkl (lzma-compressed pickle) if the fast version doesn't exist.
    """
    cache_str = str(cache_path)
    if cache_str.endswith("metric_cache.pkl"):
        fast_path = cache_str.replace("metric_cache.pkl", "metric_cache_fast.pkl")
        if os.path.exists(fast_path):
            with open(fast_path, "rb") as f:
                return pickle.load(f)
    with lzma.open(cache_str, "rb") as f:
        return pickle.load(f)


def _pdm_worker(args):
    """Process pool worker: load metric cache and compute PDM scores.

    Args:
        args: (cache, traj_np)
            cache: file path (str/Path) or pre-loaded MetricCache object
            traj_np: candidate trajectories (G, 8, 3)
    Returns:
        (scores, subscores): pairwise scores (G,) and the full sub-metrics dict.
    """
    cache, traj_np = args
    if isinstance(cache, (str, Path)):
        metric_cache = _load_metric_cache(cache)
    else:
        metric_cache = cache
    results, sim_traj = pdm_score_para(
        metric_cache=metric_cache,
        model_trajectory=traj_np,
        future_sampling=SIMULATOR.proposal_sampling,
        simulator=SIMULATOR,
        scorer=SCORER,
    )
    scores = _pairwise_scores(SCORER)
    subscores = _pairwise_subscores_full(SCORER)
    return scores.astype(np.float32), subscores


def _pairwise_subscores_ttc_ep(scorer):
    """Extract only TTC time and normalized ego progress from scorer."""
    prog_raw = scorer._progress_raw  # (N,)
    thresh = scorer._config.progress_distance_threshold

    # Normalize progress: each candidate vs GT (index 0), no multi_metrics multiplication
    raw_prog_gt = prog_raw[0]
    max_pair = np.maximum(raw_prog_gt, prog_raw[1:])
    norm_prog = np.where(
        max_pair > thresh,
        prog_raw[1:] / (max_pair + 1e-6),
        1.0,
    ).astype(np.float32)

    return {
        "ttc_time": scorer._per_step_ttc[1:, 1:],  # (N-1, 40), float32
        "ego_progress": norm_prog,                    # (N-1,), float32
    }


def _pdm_worker_ttc_ep(args):
    """Process pool worker: compute only TTC + ego progress (no collision/comfort/DDC).

    Uses GT trajectory (from transfuser_target.gz) as the reference for
    ego progress normalization instead of PDM-simulated trajectory.

    Args:
        args: (cache_path, traj_np, token_dir)
            cache_path: path to metric_cache.pkl
            traj_np: vocab trajectories (N, 8, 3)
            token_dir: path to token directory containing transfuser_target.gz
    Returns:
        subscores dict with 'ttc_time' and 'ego_progress'
    """
    cache, traj_np, token_dir = args
    if isinstance(cache, (str, Path)):
        metric_cache = _load_metric_cache(cache)
    else:
        metric_cache = cache

    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
    from navsim.planning.training.dataset import load_feature_target_from_pickle

    initial_ego_state = metric_cache.ego_state

    # Load GT trajectory from transfuser_target.gz and use as reference (index 0)
    target = load_feature_target_from_pickle(Path(os.path.join(token_dir, "transfuser_target.gz")))
    gt_traj_np = target["trajectory"].numpy() if hasattr(target["trajectory"], "numpy") else target["trajectory"]
    gt_traj = Trajectory(gt_traj_np)  # (8, 3) ego-frame
    gt_world = transform_trajectory(gt_traj, initial_ego_state)
    gt_states = get_trajectory_as_array(
        gt_world, SIMULATOR.proposal_sampling, initial_ego_state.time_point,
    )

    pred_list = []
    for i in range(traj_np.shape[0]):
        traj = Trajectory(traj_np[i])
        pred_world = transform_trajectory(traj, initial_ego_state)
        pred_states = get_trajectory_as_array(
            pred_world, SIMULATOR.proposal_sampling, initial_ego_state.time_point,
        )
        pred_list.append(pred_states)
    pred_batch = np.stack(pred_list, axis=0)

    trajectory_states = np.concatenate([gt_states[None], pred_batch], axis=0)
    simulated_states = SIMULATOR.simulate_proposals(trajectory_states, initial_ego_state)

    SCORER.score_proposals_ttc_ep(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
    )

    return _pairwise_subscores_ttc_ep(SCORER)


def _init_pool(sim_cfg, scorer_cfg, ttc_bound=1.5, use_history_comfort=False, driving_direction_exclude_intersection=False):
    """Initialize process pool globals SIMULATOR and SCORER (TrainPDMScorer).

    Uses TrainPDMScorer which extends the original PDMScorer with per-step continuous TTC
    and ego-area extraction used by reward caching / scorer training.
    """
    global SIMULATOR, SCORER
    SIMULATOR = instantiate(sim_cfg)
    from navsim.agents.flowr2a.pdm_scorer_train import PDMScorer as TrainPDMScorer
    orig_scorer = instantiate(scorer_cfg)
    SCORER = TrainPDMScorer(orig_scorer.proposal_sampling, orig_scorer._config, orig_scorer._vehicle_parameters, ttc_bound=ttc_bound, use_history_comfort=use_history_comfort, driving_direction_exclude_intersection=driving_direction_exclude_intersection)


