import numpy as np
import numpy.typing as npt

from typing import List

from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    _get_fixed_timesteps,
    _se2_vel_acc_to_ego_state,
)

from navsim.common.dataclasses import PDMResults, Trajectory
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_states_to_state_array
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex
from navsim.planning.metric_caching.metric_cache import MetricCache


def transform_trajectory(pred_trajectory: Trajectory, initial_ego_state: EgoState) -> InterpolatedTrajectory:
    """
    Transform trajectory in global frame and return as InterpolatedTrajectory
    :param pred_trajectory: trajectory dataclass in ego frame
    :param initial_ego_state: nuPlan's ego state object
    :return: nuPlan's InterpolatedTrajectory
    """

    future_sampling = pred_trajectory.trajectory_sampling
    timesteps = _get_fixed_timesteps(initial_ego_state, future_sampling.time_horizon, future_sampling.interval_length)

    relative_poses = np.array(pred_trajectory.poses, dtype=np.float64)
    relative_states = [StateSE2.deserialize(pose) for pose in relative_poses]
    absolute_states = relative_to_absolute_poses(initial_ego_state.rear_axle, relative_states)

    # NOTE: velocity and acceleration ignored by LQR + bicycle model
    agent_states = [
        _se2_vel_acc_to_ego_state(
            state,
            [0.0, 0.0],
            [0.0, 0.0],
            timestep,
            initial_ego_state.car_footprint.vehicle_parameters,
        )
        for state, timestep in zip(absolute_states, timesteps)
    ]

    # NOTE: maybe make addition of initial_ego_state optional
    return InterpolatedTrajectory([initial_ego_state] + agent_states)


def get_trajectory_as_array(
    trajectory: InterpolatedTrajectory,
    future_sampling: TrajectorySampling,
    start_time: TimePoint,
) -> npt.NDArray[np.float64]:
    """
    Interpolated trajectory and return as numpy array
    :param trajectory: nuPlan's InterpolatedTrajectory object
    :param future_sampling: Sampling parameters for interpolation
    :param start_time: TimePoint object of start
    :return: Array of interpolated trajectory states.
    """

    times_s = np.arange(
        0.0,
        future_sampling.time_horizon + future_sampling.interval_length,
        future_sampling.interval_length,
    )
    times_s += start_time.time_s
    times_us = [int(time_s * 1e6) for time_s in times_s]
    times_us = np.clip(times_us, trajectory.start_time.time_us, trajectory.end_time.time_us)
    time_points = [TimePoint(time_us) for time_us in times_us]

    trajectory_ego_states: List[EgoState] = trajectory.get_state_at_times(time_points)

    return ego_states_to_state_array(trajectory_ego_states)


def pdm_score(
    metric_cache: MetricCache,
    model_trajectory: Trajectory,
    future_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    scorer: PDMScorer,
) -> PDMResults:
    """
    Runs PDM-Score and saves results in dataclass.
    :param metric_cache: Metric cache dataclass
    :param model_trajectory: Predicted trajectory in ego frame.
    :return: Dataclass of PDM-Subscores.
    """

    initial_ego_state = metric_cache.ego_state

    pdm_trajectory = metric_cache.trajectory
    pred_trajectory = transform_trajectory(model_trajectory, initial_ego_state)

    pdm_states, pred_states = (
        get_trajectory_as_array(pdm_trajectory, future_sampling, initial_ego_state.time_point),
        get_trajectory_as_array(pred_trajectory, future_sampling, initial_ego_state.time_point),
    )

    trajectory_states = np.concatenate([pdm_states[None, ...], pred_states[None, ...]], axis=0)

    simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)

    scores = scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        **({'human_past_trajectory': getattr(metric_cache, 'past_human_trajectory', None)}
           if getattr(scorer, '_use_history_comfort', False) else {}),
    )

    # TODO: Refactor & add / modify existing metrics.
    pred_idx = 1

    no_at_fault_collisions = scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, pred_idx]
    drivable_area_compliance = scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, pred_idx]

    ego_progress = scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, pred_idx]
    time_to_collision_within_bound = scorer._weighted_metrics[WeightedMetricIndex.TTC, pred_idx]
    comfort = scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, pred_idx]
    driving_direction_compliance = scorer._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION, pred_idx]

    score = scores[pred_idx]

    return PDMResults(
        no_at_fault_collisions,
        drivable_area_compliance,
        ego_progress,
        time_to_collision_within_bound,
        comfort,
        driving_direction_compliance,
        score,
    )


from typing import Sequence, Union, List
import numpy as np

def _pdm_score_single(
    metric_cache: MetricCache,
    model_trajectory: Trajectory,
    future_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    scorer: PDMScorer,
) -> PDMResults:
    """原先的单条实现，代码完全不变"""
    initial_ego_state = metric_cache.ego_state

    pdm_trajectory = metric_cache.trajectory
    pred_trajectory = transform_trajectory(model_trajectory, initial_ego_state)

    pdm_states, pred_states = (
        get_trajectory_as_array(pdm_trajectory,
                                future_sampling,
                                initial_ego_state.time_point),
        get_trajectory_as_array(pred_trajectory,
                                future_sampling,
                                initial_ego_state.time_point),
    )

    trajectory_states = np.concatenate([pdm_states[None, ...],
                                        pred_states[None, ...]],
                                       axis=0)

    simulated_states = simulator.simulate_proposals(
        trajectory_states, initial_ego_state)

    scores = scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        **({'human_past_trajectory': getattr(metric_cache, 'past_human_trajectory', None)}
           if getattr(scorer, '_use_history_comfort', False) else {}),
    )

    # ----- 从 scorer 中取回各项指标 ----------
    pred_idx = 1
    no_at_fault_collisions     = scorer._multi_metrics[MultiMetricIndex.NO_COLLISION,    pred_idx]
    drivable_area_compliance   = scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA,    pred_idx]
    ego_progress               = scorer._weighted_metrics[WeightedMetricIndex.PROGRESS,   pred_idx]
    time_to_collision_in_bound = scorer._weighted_metrics[WeightedMetricIndex.TTC,        pred_idx]
    comfort                    = scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE,pred_idx]
    driving_dir_compliance     = scorer._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION, pred_idx]
    score                      = scores[pred_idx]

    return PDMResults(
        no_at_fault_collisions,
        drivable_area_compliance,
        ego_progress,
        time_to_collision_in_bound,
        comfort,
        driving_dir_compliance,
        score,
    )

# ---- 新的“批量 + 单条”统一入口 ---------------------------------------------
def pdm_score_para(
    metric_cache: MetricCache,
    model_trajectory: Union[
        Trajectory,
        Sequence[Trajectory],
        np.ndarray,               # (N, T+1, 3)
    ],
    future_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    scorer: PDMScorer,
) -> Union[PDMResults, List[PDMResults]]:
    """
    * 若 `model_trajectory` 是 **单条** Trajectory → 返回 PDMResults
    * 若 `model_trajectory` 是 **批量** (list / ndarray) → 返回 List[PDMResults]，长度 = N
    """
    # --------- 单条情况：直接走旧实现 ----------
    if isinstance(model_trajectory, Trajectory):
        return _pdm_score_single(metric_cache,
                                 model_trajectory,
                                 future_sampling,
                                 simulator,
                                 scorer)

    # --------- 解析批量输入 -------------------
    if isinstance(model_trajectory, np.ndarray):
        # ndarray ⇒ list[Trajectory]，假设最后两维是 (T+1, 3)
        traj_list = [Trajectory(model_trajectory[i])
                     for i in range(model_trajectory.shape[0])]
    else:
        # 已经是 list[Trajectory]
        traj_list = list(model_trajectory)

    N = len(traj_list)
    if N == 0:
        raise ValueError("Empty trajectory batch.")

    initial_ego_state = metric_cache.ego_state
    pdm_trajectory    = metric_cache.trajectory
    pdm_states        = get_trajectory_as_array(
        pdm_trajectory, future_sampling, initial_ego_state.time_point)

    # —— 把 N 条预测轨迹变成 (N, T+1, dim) ----------
    pred_states_batch = []
    for traj in traj_list:
        pred_world = transform_trajectory(traj, initial_ego_state)
        pred_states = get_trajectory_as_array(
            pred_world, future_sampling, initial_ego_state.time_point)
        pred_states_batch.append(pred_states)
    pred_states_batch = np.stack(pred_states_batch, axis=0)      # (N, T+1, dim)

    # —— 拼成 (N+1, T+1, dim)，第 0 条是 pdm traj ----------
    trajectory_states = np.concatenate(
        [pdm_states[None, ...], pred_states_batch], axis=0)

    # —— 一次仿真 + 一次打分 ------------------------
    simulated_states = simulator.simulate_proposals(
        trajectory_states, initial_ego_state)

    scores = scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        **({'human_past_trajectory': getattr(metric_cache, 'past_human_trajectory', None)}
           if getattr(scorer, '_use_history_comfort', False) else {}),
    )

    # --------- 把每条预测的指标拆出来 ----------
    results: List[PDMResults] = []
    for i in range(1, N + 1):               # 预测轨迹索引从 1 开始
        results.append(
            PDMResults(
                scorer._multi_metrics[MultiMetricIndex.NO_COLLISION,        i],
                scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA,       i],
                scorer._weighted_metrics[WeightedMetricIndex.PROGRESS,      i],
                scorer._weighted_metrics[WeightedMetricIndex.TTC,           i],
                scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE,   i],
                scorer._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION, i],
                scores[i],
            )
        )
    # sim_pred = trajectory_states[1:]
    sim_pred = simulated_states[1:]
    return results, sim_pred