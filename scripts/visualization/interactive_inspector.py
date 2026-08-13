#!/usr/bin/env python3
"""Interactive Gradio inspector for FlowR2A agent outputs.

Usage:
    python scripts/visualization/interactive_inspector.py --port 7860

    # Or pre-load an agent at startup:
    python scripts/visualization/interactive_inspector.py \
        --exp_id reward_dp_agent_v7-1_fix20_scorer_fix_all \
        --port 7860

`--exp_id` is an experiment directory under $NAVSIM_EXP_ROOT (used for checkpoint
discovery); `--agent_config` is the Hydra agent config name (default
flowr2a_agent_s2, the scorer agent).
"""

import argparse
import glob
import io
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, load_feature_target_from_pickle
from navsim.visualization.bev import add_trajectory_to_bev_ax
from navsim.visualization.config import TRAJECTORY_CONFIG
from navsim.visualization.plots import plot_bev_frame

from PIL import Image

# Inference defaults, aligned with navsim/agents/flowr2a/flowr2a_config.py
DEFAULTS = {
    "test_score_min": 0.9,
    "test_score_max": 1.0,
    "test_num_traj_sampling": 60,
    "test_weight_ttc": 1.0,
    "test_weight_ep": 1.0,
    "test_weight_c": 1.0,
    "init_step_min": 10,
    "init_step_max": 18,
}

DEFAULT_AGENT_CONFIG = "flowr2a_agent_s2"

# Pre-compute time-order colors for 8 timesteps
_TIME_CMAP = cm.get_cmap("viridis")
_TIME_COLORS_8 = [_TIME_CMAP(i / 7) for i in range(8)]


def _draw_time_markers(ax, traj, marker_size=30, marker_type="o"):
    """Draw markers on a trajectory (8,3) colored by time order (viridis)."""
    for t in range(len(traj)):
        ax.scatter(
            traj[t, 1], traj[t, 0],
            c=[_TIME_COLORS_8[t]], s=marker_size, marker=marker_type, zorder=12,
            edgecolors='black', linewidths=0.3,
        )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXP_ROOT = os.environ["NAVSIM_EXP_ROOT"]
FEATURE_CACHE = os.path.join(EXP_ROOT, "testing_cache")
DEFAULT_CLIP_INDEX = "scripts/visualization/navtest_clips_index.pkl"

OPENSCENE_ROOT = Path(os.environ["OPENSCENE_DATA_ROOT"])
NAVSIM_LOG_PATH = OPENSCENE_ROOT / "navsim_logs" / "test"
SENSOR_BLOBS_PATH = OPENSCENE_ROOT / "sensor_blobs" / "test"

os.environ["EVAL"] = "1"


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
agent: Optional[AbstractAgent] = None
dataset: Optional[CacheOnlyDataset] = None
clips_sorted: List[Dict] = []
_current: Dict[str, Any] = {}
_agent_info: Dict[str, str] = {}  # exp_id, checkpoint, agent_config


# ---------------------------------------------------------------------------
# Checkpoint discovery (from scripts/training/get_checkpoint_path.py)
# ---------------------------------------------------------------------------

def _filter_timestamp_dirs(dir_list):
    pattern = re.compile(r'^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}.*$')
    return [d for d in dir_list if pattern.match(d)]


def discover_checkpoints(exp_id: str) -> List[str]:
    """Find all checkpoints for an exp_id, newest first."""
    base_dir = os.path.join(EXP_ROOT, exp_id)
    if not os.path.isdir(base_dir):
        return []

    runs = sorted([
        d for d in _filter_timestamp_dirs(os.listdir(base_dir))
        if os.path.isdir(os.path.join(base_dir, d))
    ], reverse=True)

    ckpts = []
    for run in runs:
        ckpt_dir = os.path.join(base_dir, run, "lightning_logs", "version_0", "checkpoints")
        if not os.path.isdir(ckpt_dir):
            continue
        files = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")),
                       key=os.path.getmtime, reverse=True)
        ckpts.extend(files)
    return ckpts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_trajectory_head():
    """Get the trajectory head/decoder from the agent model, supporting both architectures."""
    model = agent._transfuser_model
    if hasattr(model, '_trajectory_head'):
        return model._trajectory_head
    elif hasattr(model, '_trajectory_decoder'):
        return model._trajectory_decoder
    else:
        raise AttributeError(f"{type(model).__name__} has no _trajectory_head or _trajectory_decoder")


def fig_to_pil(fig, dpi=120, compact=False) -> Image.Image:
    buf = io.BytesIO()
    if compact:
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    else:
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return img


def get_log_name(token: str) -> str:
    token_path = dataset._valid_cache_paths[token]
    return token_path.parent.name


def load_scene_for_token(token: str) -> Any:
    log_name = get_log_name(token)
    sf = SceneFilter(
        tokens=[token], log_names=[log_name],
        num_history_frames=4, num_future_frames=10, frame_interval=1,
    )
    sl = SceneLoader(
        data_path=NAVSIM_LOG_PATH, sensor_blobs_path=SENSOR_BLOBS_PATH,
        scene_filter=sf, sensor_config=SensorConfig.build_no_sensors(),
    )
    return sl.get_scene_from_token(token)


def render_bev(scene, trajectories=None, scores=None, best_idx=None,
               top_k=None, show_gt=True, dpi=120, traj_lw=1.0,
               color_mode="absolute", cmap_name="RdYlGn",
               show_markers=False, marker_size=30, markers_only=False,
               marker_type="o", compact=False) -> Image.Image:
    frame_idx = scene.scene_metadata.num_history_frames - 1
    fig, ax = plot_bev_frame(scene, frame_idx)

    if show_gt:
        gt = scene.get_future_trajectory(num_trajectory_frames=8)
        add_trajectory_to_bev_ax(ax, gt, TRAJECTORY_CONFIG["human"])

    if trajectories is not None and scores is not None:
        valid_mask = ~np.isnan(scores)
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) == 0:
            return fig_to_pil(fig, dpi=dpi, compact=compact)

        sorted_idx = valid_idx[np.argsort(scores[valid_idx])]
        if top_k is not None and top_k > 0 and top_k < len(sorted_idx):
            sorted_idx = sorted_idx[-top_k:]

        cmap = cm.get_cmap(cmap_name)
        if color_mode == "minmax":
            vmin, vmax = scores[valid_idx].min(), scores[valid_idx].max()
        else:
            vmin, vmax = None, None

        for idx in sorted_idx:
            traj = trajectories[idx]
            poses = np.vstack([[0, 0, 0], traj])
            if color_mode == "minmax":
                norm = 0.5 if vmax - vmin < 1e-6 else (scores[idx] - vmin) / (vmax - vmin)
            else:
                norm = scores[idx]
            color = cmap(np.clip(norm, 0, 1))
            if not markers_only:
                ax.plot(poses[:, 1], poses[:, 0], color=color,
                        linewidth=traj_lw, alpha=0.6, zorder=10)
            if show_markers:
                _draw_time_markers(ax, traj, marker_size=marker_size,
                                   marker_type=marker_type)

    return fig_to_pil(fig, dpi=dpi, compact=compact)


def batch_single(features, targets):
    bf, bt = {}, {}
    for k, v in features.items():
        if isinstance(v, torch.Tensor):
            bf[k] = v.unsqueeze(0).cuda()
        elif k == "token":
            bf[k] = [v] if v is not None else [None]
        else:
            bf[k] = v
    for k, v in targets.items():
        if isinstance(v, torch.Tensor):
            bt[k] = v.unsqueeze(0).cuda()
        else:
            bt[k] = v
    return bf, bt


# ---------------------------------------------------------------------------
# Feature visualization
# ---------------------------------------------------------------------------

def load_and_render_features(token: str, dpi=120) -> Tuple[Optional[Image.Image], Optional[Image.Image], str]:
    """Load cached transfuser_feature.gz and render lidar BEV + camera image.

    LiDAR feature shape is (C, 256, 256) where C = num_frames * 2 (below + above ground).
    Each frame is rendered as one subplot with below-ground in blue and above-ground in red.
    """
    token_path = dataset._valid_cache_paths[token]
    feature_path = token_path / "transfuser_feature.gz"
    if not feature_path.exists():
        return None, None, f"Feature cache not found: {feature_path}"

    data = load_feature_target_from_pickle(feature_path)

    images = {}
    info_lines = []

    # Camera feature: (3, H, W) float [0,1] -> RGB image
    if "camera_feature" in data:
        cam = data["camera_feature"]
        if isinstance(cam, torch.Tensor):
            cam = cam.numpy()
        cam_img = (np.clip(cam.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
        images["camera"] = Image.fromarray(cam_img)
        info_lines.append(f"Camera: {cam.shape}")

    # LiDAR feature: (C, 256, 256)
    # With ground plane: C=8 -> 4 frames * 2 layers (below, above)
    # Without: C=4 -> 4 frames * 1 layer (above only)
    if "lidar_feature" in data:
        lidar = data["lidar_feature"]
        if isinstance(lidar, torch.Tensor):
            lidar = lidar.numpy()
        n_channels = lidar.shape[0]
        grid_size = lidar.shape[1]  # 256
        info_lines.append(f"LiDAR: {lidar.shape}")

        num_frames = 4
        layers_per_frame = n_channels // num_frames
        use_ground = layers_per_frame == 2

        fig, axes = plt.subplots(1, num_frames, figsize=(4 * num_frames, 4), squeeze=False)

        color_below = "#4e79a7"  # blue
        color_above = "#e15759"  # red

        for f_idx in range(num_frames):
            ax = axes[0][f_idx]
            frame_label = f"t-{num_frames - 1 - f_idx}" if f_idx < num_frames - 1 else "t (current)"

            ax.set_facecolor("white")
            ax.set_xlim(0, grid_size)
            ax.set_ylim(0, grid_size)
            ax.set_aspect("equal")

            stats_parts = []
            if use_ground:
                ch_below = f_idx * 2
                ch_above = f_idx * 2 + 1
                bev_below = lidar[ch_below]
                bev_above = lidar[ch_above]

                nz_x, nz_y = np.where(bev_below > 0)
                if len(nz_x) > 0:
                    vals = bev_below[nz_x, nz_y]
                    ax.scatter(nz_y, nz_x, c=color_below, s=1.5,
                               alpha=np.clip(vals, 0.2, 1.0), edgecolors="none", zorder=1)
                n_below = int((bev_below > 0).sum())

                nz_x, nz_y = np.where(bev_above > 0)
                if len(nz_x) > 0:
                    vals = bev_above[nz_x, nz_y]
                    ax.scatter(nz_y, nz_x, c=color_above, s=1.5,
                               alpha=np.clip(vals, 0.2, 1.0), edgecolors="none", zorder=2)
                n_above = int((bev_above > 0).sum())

                stats_parts.append(f"below:{n_below} above:{n_above}")
            else:
                ch_above = f_idx
                bev_above = lidar[ch_above]
                nz_x, nz_y = np.where(bev_above > 0)
                if len(nz_x) > 0:
                    vals = bev_above[nz_x, nz_y]
                    ax.scatter(nz_y, nz_x, c=color_above, s=1.5,
                               alpha=np.clip(vals, 0.2, 1.0), edgecolors="none", zorder=2)
                n_above = int((bev_above > 0).sum())
                stats_parts.append(f"pts:{n_above}")

            ax.set_title(f"{frame_label}\n{' '.join(stats_parts)}", fontsize=9)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        if use_ground:
            legend_elements = [
                Patch(facecolor=color_below, label='Below ground'),
                Patch(facecolor=color_above, label='Above ground'),
            ]
            axes[0][-1].legend(handles=legend_elements, loc='lower right', fontsize=7)

        fig.suptitle("LiDAR BEV per frame ([-32m, 32m])", fontsize=12)
        fig.tight_layout()
        images["lidar"] = fig_to_pil(fig, dpi=dpi)

    # Status feature
    if "status_feature" in data:
        status = data["status_feature"]
        if isinstance(status, torch.Tensor):
            status = status.numpy()
        info_lines.append(f"Status: {status.shape}")

    info = f"Feature cache: {token_path}\n" + "\n".join(info_lines)
    return images.get("lidar"), images.get("camera"), info


# ---------------------------------------------------------------------------
# Agent loading
# ---------------------------------------------------------------------------

def _init_agent_internal(exp_id: str, checkpoint: str, agent_config: str = DEFAULT_AGENT_CONFIG):
    global agent, dataset, clips_sorted, _agent_info

    # Clear previous Hydra state for re-initialization
    GlobalHydra.instance().clear()

    config_dir = os.path.join(
        os.environ.get("NAVSIM_DEVKIT_ROOT", "."),
        "navsim/planning/script/config/pdm_scoring",
    )
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="default_run_pdm_score", overrides=[
            f"agent={agent_config}",
            f"agent.checkpoint_path='{checkpoint}'",
        ])

    new_agent = instantiate(cfg.agent)
    new_agent.initialize()
    new_agent.eval()
    new_agent.cuda()

    # Free old agent
    if agent is not None:
        agent.cpu()
        del agent
        torch.cuda.empty_cache()

    agent = new_agent
    _agent_info["exp_id"] = exp_id
    _agent_info["checkpoint"] = checkpoint
    _agent_info["agent_config"] = agent_config

    # Rebuild dataset with new agent's feature builders
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    dataset = CacheOnlyDataset(
        cache_path=FEATURE_CACHE,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        log_names=scene_filter.log_names,
        train=False,
    )

    # Rebuild clip index
    if os.path.exists(DEFAULT_CLIP_INDEX):
        with open(DEFAULT_CLIP_INDEX, "rb") as f:
            ci = pickle.load(f)
        valid_tokens = set(dataset.tokens)
        filtered = []
        for c in ci["clips"]:
            tokens = [t for t in c["tokens"] if t in valid_tokens]
            if tokens:
                filtered.append({**c, "tokens": tokens, "num_frames": len(tokens)})
        clips_sorted.clear()
        clips_sorted.extend(sorted(filtered, key=lambda c: c["num_frames"], reverse=True))

    _current.clear()
    print(f"Loaded agent {agent_config} ({exp_id}) | {len(dataset.tokens)} tokens | {len(clips_sorted)} clips")


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

def detect_checkpoints_callback(exp_id_text: str):
    """Detect checkpoints for the given exp_id and return dropdown choices."""
    exp_id = exp_id_text.strip()
    if not exp_id:
        return gr.update(choices=[], value=None), "Enter an exp_id"
    ckpts = discover_checkpoints(exp_id)
    if not ckpts:
        return gr.update(choices=[], value=None), f"No checkpoints found in {EXP_ROOT}/{exp_id}"
    # Show shortened paths for readability
    labels = []
    for p in ckpts:
        # Extract: timestamp/.../epoch=X-step=Y.ckpt
        parts = p.split(exp_id + "/")[-1] if exp_id in p else p
        labels.append(parts)
    return gr.update(choices=list(zip(labels, ckpts)), value=ckpts[0]), f"Found {len(ckpts)} checkpoint(s)"


def load_agent_callback(exp_id_text: str, checkpoint_path: str, agent_config_text: str):
    """Load agent with given exp_id and checkpoint."""
    exp_id = exp_id_text.strip()
    agent_config = agent_config_text.strip() or DEFAULT_AGENT_CONFIG
    if not checkpoint_path:
        return "Error: no checkpoint selected", *_get_head_defaults(), gr.update()

    try:
        _init_agent_internal(exp_id, checkpoint_path, agent_config)
    except Exception as e:
        return f"Error loading agent: {e}", *_get_head_defaults(), gr.update()

    info = (
        f"Loaded config: {agent_config}\n"
        f"Exp: {exp_id or '(none)'}\n"
        f"Checkpoint: {os.path.basename(checkpoint_path)}\n"
        f"Tokens: {len(dataset.tokens)} | Clips: {len(clips_sorted)}"
    )
    clip_max = max(len(clips_sorted) - 1, 0)
    return info, *_get_head_values(), gr.update(maximum=clip_max, value=0)


def _get_head_defaults():
    """Return default values when no agent is loaded."""
    return (
        DEFAULTS["test_num_traj_sampling"],  # num_traj
        DEFAULTS["init_step_min"],            # init_step_min
        DEFAULTS["init_step_max"],            # init_step_max
        DEFAULTS["test_score_min"],           # score_min
        DEFAULTS["test_score_max"],           # score_max
        DEFAULTS["test_weight_ttc"],          # w_ttc
        DEFAULTS["test_weight_ep"],           # w_ep
        DEFAULTS["test_weight_c"],            # w_c
    )


def _get_head_values():
    """Read current values from the loaded agent's trajectory head."""
    if agent is None:
        return _get_head_defaults()
    head = get_trajectory_head()
    return (
        head.test_num_traj_sampling,
        head.init_step_min,
        head.init_step_max,
        head.test_score_min,
        head.test_score_max,
        head.test_weight_ttc,
        head.test_weight_ep,
        head.test_weight_c,
    )


def resolve_token(token_text: str, clip_idx: int, frame_in_clip: int) -> Optional[str]:
    token_text = token_text.strip()
    if token_text:
        return token_text if token_text in dataset._valid_cache_paths else None
    if clip_idx < 0 or clip_idx >= len(clips_sorted):
        return None
    clip = clips_sorted[clip_idx]
    tokens_in_clip = [t for t in clip["tokens"] if t in dataset._valid_cache_paths]
    if frame_in_clip < 0 or frame_in_clip >= len(tokens_in_clip):
        return None
    return tokens_in_clip[frame_in_clip]


def load_scene_callback(token_text, clip_idx, frame_in_clip, dpi, show_gt, compact):
    if agent is None:
        return None, "No agent loaded. Load an agent first."
    token = resolve_token(token_text, clip_idx, frame_in_clip)
    if token is None:
        return None, "Token not found in dataset."

    scene = load_scene_for_token(token)
    features, targets = dataset._load_scene_with_token(token)

    _current["token"] = token
    _current["scene"] = scene
    _current["features"] = features
    _current["targets"] = targets

    log_name = get_log_name(token)
    bev = render_bev(scene, show_gt=show_gt, dpi=int(dpi), compact=compact)
    info = f"Token: {token}\nLog: {log_name}\nDataset tokens: {len(dataset.tokens)}"
    return bev, info


def run_inference_callback(
    num_traj, init_step_min, init_step_max,
    score_min, score_max,
    w_ttc, w_ep, w_c,
    top_k, dpi, traj_lw, color_mode_val,
    show_gt, marker_mode, marker_size, color_theme, marker_type,
    compact,
):
    if agent is None:
        return None, "No agent loaded."
    if "features" not in _current:
        return None, "No scene loaded. Load a scene first."

    head = get_trajectory_head()

    # Apply inference settings
    head.test_num_traj_sampling = int(num_traj)
    head.init_step_min = int(init_step_min)
    head.init_step_max = int(init_step_max)
    head.test_score_min = float(score_min)
    head.test_score_max = float(score_max)
    head.test_weight_ttc = float(w_ttc)
    head.test_weight_ep = float(w_ep)
    head.test_weight_c = float(w_c)

    bf, bt = batch_single(_current["features"], _current["targets"])
    with torch.no_grad():
        out = agent.forward(bf, bt, multi_return=True)

    trajs = out["trajs"][0].cpu().numpy()
    scores = out["scores"][0].cpu().numpy()
    best = int(out["best_idx"][0].cpu().item())
    subscore = out.get("subscore", {})

    # Recompute display scores from subscores: NC*DAC*(w1*TTC+w2*EP+w3*C)/denom
    if all(k in subscore for k in ["NC", "DAC", "TTC", "EP", "C"]):
        nc = subscore["NC"][0].cpu().numpy()   # (K,)
        dac = subscore["DAC"][0].cpu().numpy()
        ttc = subscore["TTC"][0].cpu().numpy()
        ep = subscore["EP"][0].cpu().numpy()
        c = subscore["C"][0].cpu().numpy()
        denom = float(w_ttc) + float(w_ep) + float(w_c)
        if denom < 1e-6:
            denom = 1.0
        display_scores = nc * dac * (float(w_ttc) * ttc + float(w_ep) * ep + float(w_c) * c) / denom
    else:
        display_scores = scores

    best_score = display_scores[best] if not np.isnan(display_scores[best]) else 0.0
    lines = [
        f"Trajectories: {len(trajs)}",
        f"Best idx: {best}  Score: {best_score:.4f}",
        f"Score range: [{np.nanmin(display_scores):.4f}, {np.nanmax(display_scores):.4f}]",
        f"Mean score: {np.nanmean(display_scores):.4f}",
        "",
        "Subscores (best traj):",
    ]
    for name in ["NC", "DAC", "TTC", "EP", "C"]:
        if name in subscore:
            val = subscore[name][0, best].cpu().item()
            lines.append(f"  {name}: {val:.4f}")
    lines.extend([
        "",
        f"Agent: {_agent_info.get('agent_config', '?')} ({_agent_info.get('exp_id', '?')})",
        f"Settings: n={int(num_traj)} "
        f"init_step=[{int(init_step_min)},{int(init_step_max)}] "
        f"score=[{score_min:.2f},{score_max:.2f}]",
    ])

    top_k_val = int(top_k) if top_k and top_k > 0 else None
    _show_markers = marker_mode != "off"
    _markers_only = marker_mode == "markers only"
    cmap_for_render = color_theme if color_theme != "categorical" else "RdYlGn"
    bev = render_bev(_current["scene"], trajs, display_scores, best, top_k=top_k_val,
                     show_gt=show_gt,
                     dpi=int(dpi), traj_lw=float(traj_lw),
                     color_mode=color_mode_val, cmap_name=cmap_for_render,
                     show_markers=_show_markers, marker_size=float(marker_size),
                     markers_only=_markers_only, marker_type=marker_type,
                     compact=compact)

    # Cache for save
    _current["last_bev"] = bev
    _current["last_info"] = "\n".join(lines)
    _current["last_settings"] = {
        "exp_id": _agent_info.get("exp_id", ""),
        "agent_config": _agent_info.get("agent_config", ""),
        "checkpoint": _agent_info.get("checkpoint", ""),
        "num_traj_sampling": int(num_traj),
        "init_step_min": int(init_step_min),
        "init_step_max": int(init_step_max),
        "score_min": float(score_min),
        "score_max": float(score_max),
        "w_ttc": float(w_ttc),
        "w_ep": float(w_ep),
        "w_c": float(w_c),
        "top_k": top_k_val,
        "dpi": int(dpi),
        "traj_lw": float(traj_lw),
    }

    return bev, "\n".join(lines)


def save_callback(clip_idx):
    """Save current BEV image and inference config."""
    if "last_bev" not in _current or "token" not in _current:
        return "Nothing to save. Run inference first."

    token = _current["token"]
    ci = int(clip_idx) if clip_idx is not None else -1
    clip_dir = f"clip_{ci}" if ci >= 0 else "direct"
    out_dir = Path("viz_output/inspector") / clip_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find next available index
    i = 0
    while (out_dir / f"{token}_{i}.png").exists():
        i += 1

    img_path = out_dir / f"{token}_{i}.png"
    cfg_path = out_dir / f"{token}_{i}.json"

    _current["last_bev"].save(str(img_path))

    save_data = _current.get("last_settings", {})
    save_data["token"] = token
    save_data["clip_idx"] = ci
    save_data["log_name"] = get_log_name(token) if dataset else ""
    save_data["info"] = _current.get("last_info", "")
    with open(cfg_path, "w") as f:
        json.dump(save_data, f, indent=2)

    return f"Saved: {img_path}"


def get_clip_info(clip_idx):
    if not clips_sorted or clip_idx < 0 or clip_idx >= len(clips_sorted):
        return "No clips loaded", gr.update(maximum=0, value=0)
    clip = clips_sorted[clip_idx]
    valid_tokens = [t for t in clip["tokens"] if t in dataset._valid_cache_paths]
    info = f"Log: {clip['log_name']}\nFrames: {len(valid_tokens)} (raw: {clip['num_frames']})"
    return info, gr.update(maximum=max(0, len(valid_tokens) - 1), value=0)


def show_features_callback(dpi):
    """Load and display cached lidar/camera features."""
    if "token" not in _current:
        return None, None, "No scene loaded."
    try:
        lidar_img, cam_img, info = load_and_render_features(
            _current["token"], dpi=int(dpi))
    except Exception as e:
        return None, None, f"Error: {e}"
    return lidar_img, cam_img, info


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

def build_app(initial_exp_id: str = ""):
    with gr.Blocks(title="FlowR2A Inspector") as app:
        gr.Markdown("# FlowR2A Agent Inspector")

        # --- Agent Selection ---
        with gr.Accordion("Agent Selection", open=(agent is None)):
            with gr.Row():
                exp_id_input = gr.Textbox(
                    label="Experiment ID (dir under $NAVSIM_EXP_ROOT, for checkpoint discovery)",
                    value=initial_exp_id,
                    placeholder="e.g. reward_dp_agent_v7-1_fix20_scorer_fix_all",
                )
                detect_btn = gr.Button("Detect Checkpoints")
            with gr.Row():
                ckpt_dropdown = gr.Dropdown(label="Checkpoint", choices=[], interactive=True)
                detect_info = gr.Textbox(label="Status", interactive=False, lines=1)
            agent_config_input = gr.Textbox(
                label="Agent config (Hydra)", value=DEFAULT_AGENT_CONFIG,
                placeholder="flowr2a_agent_s2",
            )
            load_agent_btn = gr.Button("Load Agent", variant="primary")
            agent_info = gr.Textbox(label="Agent info", interactive=False, lines=4,
                                     value=f"Loaded: {_agent_info.get('agent_config', 'None')}" if agent else "No agent loaded")

        with gr.Row():
            # --- Left: Scene Selection ---
            with gr.Column(scale=1):
                gr.Markdown("### Scene Selection")
                token_input = gr.Textbox(label="Token (direct)", placeholder="e.g. 67628d15c5b45860")
                gr.Markdown("**--- OR select by clip ---**")
                clip_slider = gr.Slider(0, max(len(clips_sorted) - 1, 0), step=1, value=0, label="Clip index")
                clip_info_text = gr.Textbox(label="Clip info", interactive=False, lines=2)
                frame_slider = gr.Slider(0, 50, step=1, value=0, label="Frame in clip")
                load_btn = gr.Button("Load Scene", variant="primary")
                scene_info = gr.Textbox(label="Scene info", interactive=False, lines=3)

                gr.Markdown("### Render Settings")
                with gr.Row():
                    dpi_slider = gr.Slider(80, 300, step=10, value=120, label="DPI (resolution)")
                    traj_lw_slider = gr.Slider(0.2, 5.0, step=0.1, value=1.0, label="Traj line width")
                color_mode = gr.Dropdown(["absolute", "minmax"], value="absolute", label="Color mode")
                color_theme = gr.Dropdown(
                    ["categorical", "RdYlGn", "inferno", "viridis", "plasma", "magma",
                     "cividis", "coolwarm", "Spectral", "RdBu_r"],
                    value="categorical", label="Color theme",
                    info="Colormap for trajectory scores; categorical = RdYlGn",
                )
                with gr.Row():
                    show_gt = gr.Checkbox(value=True, label="Show GT trajectory")
                    compact_mode = gr.Checkbox(value=False, label="Compact (no title/margin)")
                    marker_mode = gr.Dropdown(
                        ["off", "with lines", "markers only"],
                        value="off", label="Point markers",
                        info="Markers colored by time order (dark=early, bright=late)",
                    )
                    marker_size_slider = gr.Slider(1, 100, step=1, value=30, label="Marker size")
                    marker_type_dropdown = gr.Dropdown(
                        ["o", "s", "^", "v", "D", ".", "x", "+", "*"],
                        value="o", label="Marker type",
                        info="o=circle, s=square, ^/v=triangle, D=diamond, .=point, x/+=cross, *=star",
                    )

            # --- Right: Inference Settings ---
            with gr.Column(scale=1):
                gr.Markdown("### Inference Settings")
                dv = _get_head_values() if agent else _get_head_defaults()
                num_traj = gr.Slider(1, 200, step=1, value=dv[0], label="num_traj_sampling")
                with gr.Row():
                    init_step_min = gr.Slider(0, 20, step=1, value=dv[1], label="init_step_min")
                    init_step_max = gr.Slider(0, 20, step=1, value=dv[2], label="init_step_max")
                with gr.Row():
                    score_min = gr.Slider(0, 1, step=0.01, value=dv[3], label="score_min")
                    score_max = gr.Slider(0, 1, step=0.01, value=dv[4], label="score_max")
                with gr.Row():
                    w_ttc = gr.Slider(0, 20, step=0.5, value=dv[5], label="w_ttc")
                    w_ep = gr.Slider(0, 20, step=0.5, value=dv[6], label="w_ep")
                w_c = gr.Slider(0, 20, step=0.5, value=dv[7], label="w_c")
                top_k = gr.Slider(0, 200, step=1, value=0, label="top_k (0=show all)")
                run_btn = gr.Button("Run Inference", variant="primary")

        # --- Bottom: Results ---
        with gr.Row():
            bev_image = gr.Image(label="BEV", type="pil")
            with gr.Column():
                result_text = gr.Textbox(label="Results", lines=13, interactive=False)
                with gr.Row():
                    save_btn = gr.Button("Save Image + Config")
                    save_status = gr.Textbox(label="Save status", interactive=False, lines=1)

        # --- Cached Features ---
        with gr.Accordion("Cached Features (LiDAR / Camera)", open=False):
            show_features_btn = gr.Button("Show LiDAR / Camera Features")
            with gr.Row():
                lidar_image = gr.Image(label="LiDAR BEV", type="pil")
                camera_image = gr.Image(label="Camera (stitched)", type="pil")
            feature_info_text = gr.Textbox(label="Feature info", interactive=False, lines=3)

        # --- Settings outputs for load_agent to update ---
        settings_outputs = [
            num_traj, init_step_min, init_step_max,
            score_min, score_max,
            w_ttc, w_ep, w_c,
        ]

        # --- Callbacks ---
        detect_btn.click(detect_checkpoints_callback,
                         inputs=[exp_id_input],
                         outputs=[ckpt_dropdown, detect_info])
        load_agent_btn.click(load_agent_callback,
                             inputs=[exp_id_input, ckpt_dropdown, agent_config_input],
                             outputs=[agent_info, *settings_outputs, clip_slider])
        clip_slider.change(get_clip_info, inputs=[clip_slider], outputs=[clip_info_text, frame_slider])
        load_btn.click(load_scene_callback,
                       inputs=[token_input, clip_slider, frame_slider, dpi_slider, show_gt, compact_mode],
                       outputs=[bev_image, scene_info])
        run_btn.click(run_inference_callback,
                      inputs=[num_traj, init_step_min, init_step_max,
                              score_min, score_max,
                              w_ttc, w_ep, w_c,
                              top_k, dpi_slider, traj_lw_slider, color_mode,
                              show_gt, marker_mode, marker_size_slider, color_theme,
                              marker_type_dropdown, compact_mode],
                      outputs=[bev_image, result_text])
        save_btn.click(save_callback, inputs=[clip_slider], outputs=[save_status])
        show_features_btn.click(show_features_callback,
                                inputs=[dpi_slider],
                                outputs=[lidar_image, camera_image, feature_info_text])

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive FlowR2A inspector")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment dir under $NAVSIM_EXP_ROOT (for checkpoint discovery)")
    parser.add_argument("--agent_config", type=str, default=DEFAULT_AGENT_CONFIG, help="Hydra agent config name")
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint path (auto-detected from --exp_id if empty)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    # Optionally pre-load agent at startup
    if args.exp_id or args.checkpoint:
        ckpt = args.checkpoint
        if not ckpt and args.exp_id:
            ckpts = discover_checkpoints(args.exp_id)
            if ckpts:
                ckpt = ckpts[0]
                print(f"Auto-detected checkpoint: {ckpt}")
            else:
                print(f"Warning: no checkpoints found for {args.exp_id}")
        if ckpt:
            _init_agent_internal(args.exp_id, ckpt, args.agent_config)

    app = build_app(initial_exp_id=args.exp_id)
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
