import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from typing import Tuple, Any, List, Optional

# 修正后的导入路径
from navsim.common.dataclasses import Scene, Camera
from navsim.visualization.config import CAMERAS_PLOT_CONFIG, TRAJECTORY_CONFIG
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
from navsim.visualization.camera import add_camera_ax
from navsim.visualization.plots import configure_all_ax, configure_bev_ax  #
from navsim.common.dataclasses import Trajectory
# cmap = plt.get_cmap('inferno')
#     colors = cmap(scores)
def get_colors_from_scores(scores: npt.NDArray[np.float32], colormap: str = "inferno") -> npt.NDArray[np.uint8]:
    """
    将分数映射为 RGB 颜色。
    """
    if len(scores) == 0:
        return np.array([])
    
    # 归一化分数
    # min_score, max_score = scores.min(), scores.max()
    # if max_score - min_score < 1e-6:
    #     norm_scores = np.zeros_like(scores)
    # else:
    #     norm_scores = (scores - min_score) / (max_score - min_score)
    
    cmap = plt.get_cmap(colormap)
    colors = cmap(scores)  # (N, 4) RGBA, 0-1
    return (colors[:, :3] * 255).astype(np.uint8)

def project_lidar_polyline_to_camera(
    polyline: npt.NDArray[np.float32],
    camera: Camera,
    z_height: float = 0  # 假设地面相对于 Lidar 的高度
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """
    将 Lidar 坐标系下的轨迹投影到相机平面。
    :param polyline: (T, 2) 轨迹点 x,y
    :return: (T, 2) 像素坐标, (T,) 有效性Mask(在相机前方)
    """
    # 1. 补全 Z 轴 (T, 2) -> (T, 3)
    points_3d = np.hstack([polyline, np.full((polyline.shape[0], 1), z_height)])
    
    # 2. 获取变换矩阵：Lidar -> Camera -> Image
    # 参考 camera.py 中的逻辑
    lidar2cam_r = np.linalg.inv(camera.sensor2lidar_rotation)
    lidar2cam_t = camera.sensor2lidar_translation @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    viewpad = np.eye(4)
    viewpad[: camera.intrinsics.shape[0], : camera.intrinsics.shape[1]] = camera.intrinsics
    
    # 组合变换
    lidar2img_rt = viewpad @ lidar2cam_rt.T

    # 3. 执行投影
    points_homo = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1))], axis=-1) # (T, 4)
    points_img_homo = (lidar2img_rt @ points_homo.T).T # (T, 4)

    # 4. 归一化与深度过滤
    depth = points_img_homo[:, 2]
    mask = depth > 1e-3  # 仅保留相机前方的点

    # 透视除法得到像素坐标
    points_img = points_img_homo[:, :2] / np.maximum(depth[:, None], 1e-3)
    
    return points_img, mask

def add_trajectories_to_camera_ax(
    ax: plt.Axes, 
    camera: Camera, 
    trajectories: npt.NDArray[np.float32], 
    colors: npt.NDArray[np.uint8]
) -> plt.Axes:
    """
    在相机视图上叠加绘制多条轨迹。
    """
    image = camera.image.copy()
    h, w = image.shape[:2]

    # 遍历每条轨迹
    for i, traj in enumerate(trajectories):
        # 投影
        # assert False
        pts_img, mask = project_lidar_polyline_to_camera(traj, camera, z_height=0)
        
        # 简单的过滤：只有当连续的点都在相机前方时才绘制线段
        # 这里为了视觉连贯性，如果整条轨迹大部分不可见，可能就不会画出来
        if np.sum(mask) < 2:
            continue
            
        valid_pts = pts_img[mask].astype(np.int32)
        
        # 进一步过滤：检查点是否在图像范围内（可选，OpenCV会自动裁剪，但为了性能可先过滤）
        # 这里直接交给 OpenCV 处理裁剪
        if len(valid_pts) > 1:
            color = tuple(map(int, colors[i])) # (R, G, B)
            # 使用 cv2.polylines 绘制抗锯齿线段
            cv2.polylines(image, [valid_pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

    ax.imshow(image)
    return ax

def add_trajectories_to_bev_ax(
    ax: plt.Axes,
    trajectories: npt.NDArray[np.float32],
    colors: npt.NDArray[np.uint8]
) -> plt.Axes:
    """
    在 BEV 视图上绘制多条轨迹。
    注意：navsim 的 BEV 坐标系定义中，绘图时通常使用 plot(y, x) 并反转 x 轴。
    """
    trajectories = np.concatenate([np.zeros([trajectories.shape[0], 1, 2]), trajectories[:, :, :2]], axis = 1)
    for i, traj in enumerate(trajectories):
        color_norm = colors[i] / 255.0 # matplotlib 需要 0-1 的颜色
        # print(traj)
        # 参考 bev.py 中的 add_trajectory_to_bev_ax，使用 (y, x) 进行绘制
        ax.plot(
            traj[:, 1], # Y 坐标
            traj[:, 0], # X 坐标
            color=color_norm,
            linewidth=3,
            alpha=0.8,
            zorder=10 # 确保画在最上层
        )
    return ax

def add_trajectory_to_bev_ax(ax: plt.Axes, trajectory: Trajectory, config) -> plt.Axes:
    """
    Add trajectory poses as lint to plot
    :param ax: matplotlib ax object
    :param trajectory: navsim trajectory dataclass
    :param config: dictionary with plot parameters
    :return: ax with plot
    """
    poses = np.concatenate([np.array([[0, 0]]), trajectory.poses[:, :2]])
    ax.plot(
        poses[:, 1],
        poses[:, 0],
        color=config["line_color"],
        alpha=config["line_color_alpha"],
        linewidth=config["line_width"],
        linestyle=config["line_style"],
        marker=config["marker"],
        markersize=config["marker_size"],
        markeredgecolor=config["marker_edge_color"],
        zorder=config["zorder"],
    )
    return ax


def plot_cameras_frame_with_trajectories(
    scene: Scene, 
    frame_idx: int, 
    trajectories: npt.NDArray[np.float32], 
    scores: npt.NDArray[np.float32]
) -> Tuple[plt.Figure, Any]:
    """
    绘制 8个相机视图 + 1个 BEV 视图，并在所有视图上叠加带分数的轨迹。
    
    :param scene: Navsim Scene 对象
    :param frame_idx: 帧索引
    :param trajectories: 形状为 (N, 8, 2) 的轨迹数组 (Lidar坐标系 x,y)
    :param scores: 形状为 (N,) 的分数数组
    """
    frame = scene.frames[frame_idx]

    indices = scores.argsort()
    scores = scores[indices]
    trajectories = trajectories[indices]
    
    # 生成颜色
    colors = get_colors_from_scores(scores)
    
    # 初始化 3x3 画布
    fig, ax = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"])
    
    trajectories_xy = trajectories[...,:2]

    # 辅助函数：绘制相机底图并叠加轨迹
    def plot_cam_layer(ax_obj, cam_obj):
        # 先画底图，再画轨迹
        # 注意：add_camera_ax 只是 imshow，无法叠加，所以我们需要在图像矩阵上画完再 show
        add_trajectories_to_camera_ax(ax_obj, cam_obj, trajectories_xy, colors)

    # === 第一行 ===
    plot_cam_layer(ax[0, 0], frame.cameras.cam_l0)
    plot_cam_layer(ax[0, 1], frame.cameras.cam_f0)
    plot_cam_layer(ax[0, 2], frame.cameras.cam_r0)

    # === 第二行 ===
    plot_cam_layer(ax[1, 0], frame.cameras.cam_l1)
    
    # 中间：BEV + Map + Trajectories
    # 先画基础 BEV (Map)
    add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)
    # 再叠加轨迹
    add_trajectories_to_bev_ax(ax[1, 1], trajectories, colors)
    # for trajectory in trajectories:
    #     add_trajectory_to_bev_ax(ax[1, 1], Trajectory(trajectory), TRAJECTORY_CONFIG["agentv2"])
    
    plot_cam_layer(ax[1, 2], frame.cameras.cam_r1)

    # === 第三行 ===
    plot_cam_layer(ax[2, 0], frame.cameras.cam_l2)
    plot_cam_layer(ax[2, 1], frame.cameras.cam_b0)
    plot_cam_layer(ax[2, 2], frame.cameras.cam_r2)

    # 配置样式 (隐藏坐标轴等)
    configure_all_ax(ax)
    
    # 配置 BEV 特有的坐标范围和翻转
    configure_bev_ax(ax[1, 1]) 
    
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

    return fig, ax