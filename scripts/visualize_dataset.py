#!/usr/bin/env python3
"""
ScribeMem-Bench 数据集可视化脚本。

生成一张综合大图，展示提取数据的全貌：
  1. 机器狗2D轨迹 + 记忆节点位置（叠加在地图 occupancy 上）
  2. 采样图像缩略图（沿轨迹排列）
  3. 速度曲线（线速度 + 角速度）
  4. 激光雷达点云俯视投影
  5. 数据统计面板
"""

import csv
import glob as _glob
import json
import math
import os
import struct
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.font_manager as fm
import numpy as np

try:
    import yaml
except ImportError:
    yaml = None

# ── 中文字体配置 ──────────────────────────────────────────
_CN_FONT = None
for fp in fm.fontManager.ttflist:
    if "Noto Sans CJK SC" in fp.name or "Noto Sans CJK" in fp.name:
        _CN_FONT = fp.fname
        break
if _CN_FONT:
    matplotlib.rcParams["font.family"] = fm.FontProperties(fname=_CN_FONT).get_name()
matplotlib.rcParams["axes.unicode_minus"] = False


# ── 数据加载工具 ──────────────────────────────────────────

def load_csv(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def load_json(path: str) -> dict | list:
    with open(path) as f:
        return json.load(f)


# ── 地图加载 + 轨迹合成 ──────────────────────────────────

def _load_map_background(scenes_dir: str, map_id: str) -> tuple:
    """加载 occupancy 地图和场景定义。返回 (img, extent, rooms, map_meta)。"""
    scene_dir = os.path.join(scenes_dir, map_id)
    layout_path = os.path.join(scene_dir, "layout.png")
    scene_path = os.path.join(scene_dir, "scene.yaml")

    img = None; extent = None; rooms = []
    if os.path.exists(layout_path):
        img = cv2.imread(layout_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # ROS occupancy grid: top row = north = high Y, bottom row = south = low Y
            # matplotlib imshow with extent already maps top row → top of extent correctly
            # No flip needed — the image orientation matches the coordinate system
            h, w = img.shape
            # 尝试读 occupancy.yaml 拿 origin + resolution
            occ_yaml = os.path.join(os.path.expanduser("~/.robonix/maps"), map_id, "occupancy.yaml")
            if yaml and os.path.exists(occ_yaml):
                occ = yaml.safe_load(open(occ_yaml)) or {}
            elif yaml and os.path.exists(scene_path):
                occ = yaml.safe_load(open(scene_path)) or {}
            else:
                occ = {}
            resolution = float(occ.get("resolution", 0.05))
            origin_x, origin_y = float(occ.get("origin", [0, 0, 0])[0]), float(occ.get("origin", [0, 0, 0])[1])
            extent = (origin_x, origin_x + w * resolution, origin_y, origin_y + h * resolution)

    if yaml and os.path.exists(scene_path):
        scene = yaml.safe_load(open(scene_path)) or {}
        rooms = scene.get("rooms", [])

    return img, extent, rooms


def _compose_map_trajectory(output_dir: str) -> list[dict]:
    """从 TF 数据合成 map→base_link 轨迹。时间对齐 map→odom + odom→base_link。"""
    tf_path = os.path.join(output_dir, "tf_trajectory.csv")
    if not os.path.exists(tf_path):
        return []
    rows = list(csv.DictReader(open(tf_path)))

    # 分开两层 TF，按时间排序
    map_odom = sorted(
        [(int(r["timestamp_ns"]), float(r["tx"]), float(r["ty"]), float(r["rz"]), float(r["rw"]))
         for r in rows if r["frame_id"] == "map" and r["child_frame_id"] == "odom"],
        key=lambda x: x[0])
    odom_bl = sorted(
        [(int(r["timestamp_ns"]), float(r["tx"]), float(r["ty"]), float(r["rz"]), float(r["rw"]))
         for r in rows if r["frame_id"] == "odom" and r["child_frame_id"] == "base_link"],
        key=lambda x: x[0])

    if not map_odom:
        return []

    # 时间对齐：对每个 map→odom 找最近的 odom→base_link
    result = []
    oi = 0  # odom index
    for t_mo, mx, my, mrz, mrw in map_odom:
        # 找时间最近的 odom→base_link
        while oi + 1 < len(odom_bl) and abs(odom_bl[oi + 1][0] - t_mo) < abs(odom_bl[oi][0] - t_mo):
            oi += 1
        if oi < len(odom_bl) and abs(odom_bl[oi][0] - t_mo) < 5e8:  # 0.5s 窗口
            _, ox, oy, orz, orw = odom_bl[oi]
        else:
            ox, oy, orz, orw = 0.0, 0.0, 0.0, 1.0  # identity fallback

        # TF 组合: T_map_bl = T_map_odom * T_odom_bl
        # 2D 简化: 旋转叠加，位移旋转叠加
        cos_m, sin_m = _quat_cos_sin(mrz, mrw)
        cos_o, sin_o = _quat_cos_sin(orz, orw)
        cos_c = cos_m * cos_o - sin_m * sin_o
        sin_c = sin_m * cos_o + cos_m * sin_o
        cx = mx + cos_m * ox - sin_m * oy
        cy = my + sin_m * ox + cos_m * oy

        result.append({"tx": cx, "ty": cy, "timestamp_ns": t_mo,
                       "frame_id": "map", "child_frame_id": "base_link"})
    return result


def _quat_cos_sin(rz: float, rw: float) -> tuple:
    """从 2D 四元数 (0,0,rz,rw) 提取 cos/sin。"""
    return (1 - 2 * rz * rz), (2 * rz * rw)


def _detect_map_id(output_dir: str, project_dir: str) -> str:
    """从 session meta.yaml 或 raw meta.yaml 获取 map_id。"""
    for src in [os.path.join(output_dir, "meta.yaml"),
                os.path.join(project_dir, "raw",
                             os.path.basename(output_dir), "meta.yaml")]:
        if yaml and os.path.exists(src):
            meta = yaml.safe_load(open(src)) or {}
            mid = meta.get("map_id", "")
            if mid:
                return mid
    return ""


# ── 绘图组件 ──────────────────────────────────────────────

def plot_trajectory(ax, traj: list[dict], nodes: list[dict], images_idx: list[dict],
                    map_img=None, map_extent=None, rooms=None, map_label: str = ""):
    """绘制2D轨迹（俯视图）+ 记忆节点 + 图像采样点，可选地图背景。"""
    tx = [float(r["tx"]) for r in traj]
    ty = [float(r["ty"]) for r in traj]

    # 地图背景
    if map_img is not None and map_extent is not None:
        ax.imshow(map_img, extent=map_extent, cmap="gray", alpha=0.35, zorder=0,
                  interpolation="nearest")
        # 房间多边形
        if rooms:
            for room in rooms:
                pts = room.get("points", [])
                if pts:
                    poly = mpatches.Polygon(pts, fill=False, edgecolor="#E74C3C",
                                            linewidth=1.2, linestyle="--", alpha=0.7, zorder=1)
                    ax.add_patch(poly)
                    cx = sum(p[0] for p in pts) / len(pts)
                    cy = sum(p[1] for p in pts) / len(pts)
                    ax.text(cx, cy, room.get("name", ""), fontsize=6, color="#C0392B",
                            ha="center", va="center", alpha=0.8, zorder=2)

    # 轨迹线
    frame_label = f" [{traj[0].get('frame_id','?')}→{traj[0].get('child_frame_id','?')}]" if traj else ""
    ax.plot(tx, ty, linewidth=1.2, color="#4A90D9", alpha=0.7, zorder=3,
            label=f"轨迹{frame_label}")

    # 起点和终点
    if tx:
        ax.scatter(tx[0], ty[0], s=120, marker="o", color="#2ECC40", edgecolors="white",
                   linewidth=1.5, zorder=5, label=f"起点 ({tx[0]:.2f}, {ty[0]:.2f})")
        ax.scatter(tx[-1], ty[-1], s=120, marker="s", color="#FF4136", edgecolors="white",
                   linewidth=1.5, zorder=5, label=f"终点 ({tx[-1]:.2f}, {ty[-1]:.2f})")

    # 记忆节点
    nx = [n["spatial_data"]["position"]["x"] for n in nodes]
    ny = [n["spatial_data"]["position"]["y"] for n in nodes]
    ax.scatter(nx, ny, s=50, marker="D", color="#FF851B", edgecolors="white",
               linewidth=0.8, zorder=5, label=f"记忆节点 ({len(nodes)}个)")

    # 节点编号
    for n in nodes:
        ax.annotate(str(n["node_id"]), (n["spatial_data"]["position"]["x"],
                    n["spatial_data"]["position"]["y"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=6,
                    color="#555555")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    title = "机器狗巡检轨迹 (俯视图)"
    if map_label:
        title += f"  —  {map_label}"
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


def plot_velocity(ax, odom: list[dict]):
    """绘制线速度和角速度曲线。"""
    t0 = int(odom[0]["timestamp_ns"])
    ts = [(int(r["timestamp_ns"]) - t0) / 1e9 for r in odom]
    lin_v = [math.sqrt(float(r["lin_vel_x"])**2 + float(r["lin_vel_y"])**2) for r in odom]
    ang_v = [abs(float(r["ang_vel_z"])) for r in odom]

    ax2 = ax.twinx()
    ax.plot(ts, lin_v, linewidth=0.8, color="#4A90D9", alpha=0.8, label="线速度 (m/s)")
    ax2.plot(ts, ang_v, linewidth=0.8, color="#FF851B", alpha=0.8, label="角速度 (rad/s)")

    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("线速度 (m/s)", color="#4A90D9")
    ax2.set_ylabel("角速度 (rad/s)", color="#FF851B")
    ax.set_title("速度曲线", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)


def plot_image_strip(ax, output_dir: str, images_idx: list[dict], nodes: list[dict], max_show: int = 12):
    """沿轨迹等距选取图像缩略图，横向排列。"""
    # 选取与记忆节点关联的图像
    node_images = []
    img_map = {img["filename"]: img for img in images_idx}

    for node in nodes:
        for ref in node.get("image_refs", []):
            fname = os.path.basename(ref)
            if fname in img_map:
                node_images.append((node["node_id"], fname))
                break

    if not node_images:
        ax.text(0.5, 0.5, "无关联图像", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("巡检帧缩略图", fontweight="bold")
        return

    # 均匀采样
    step = max(1, len(node_images) // max_show)
    sampled = node_images[::step][:max_show]

    images_dir = os.path.join(output_dir, "images")
    for i, (nid, fname) in enumerate(sampled):
        img_path = os.path.join(images_dir, fname)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (120, 90))
            ax_img = ax.inset_axes([i / len(sampled), 0, 1 / len(sampled), 1], transform=ax.transAxes)
            ax_img.imshow(img)
            ax_img.axis("off")
            ax_img.set_title(f"#{nid}", fontsize=7, pad=1)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("巡检帧缩略图 (关联记忆节点)", fontweight="bold")


def plot_lidar_overhead(ax, lidar_dir: str, max_points: int = 50000):
    """绘制激光雷达点云俯视投影（从 PCD 文件读取，随机采样）。"""
    pcd_files = sorted(_glob.glob(os.path.join(lidar_dir, "*.pcd")))
    if not pcd_files:
        ax.text(0.5, 0.5, "无点云数据 (PCD)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("激光雷达点云俯视图", fontweight="bold")
        return

    # 均匀采样 PCD 文件，控制总点数
    files_to_read = pcd_files
    if len(pcd_files) > 20:
        step = max(1, len(pcd_files) // 20)
        files_to_read = pcd_files[::step]

    pts_per_file = max(1, max_points // len(files_to_read))
    xs, ys = [], []

    for fpath in files_to_read:
        with open(fpath) as f:
            in_data = False
            local_count = 0
            for line in f:
                if line.startswith("DATA"):
                    in_data = True
                    continue
                if not in_data or line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    if local_count % max(1, 10) == 0 and len(xs) < max_points:
                        xs.append(float(parts[0]))
                        ys.append(float(parts[1]))
                    local_count += 1

    if not xs:
        ax.text(0.5, 0.5, "无点云数据", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("激光雷达点云俯视图", fontweight="bold")
        return

    ax.scatter(xs, ys, s=0.3, color="#333333", alpha=0.4, rasterized=True)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"激光雷达点云俯视图 (采样 {len(xs):,} 点, {len(pcd_files)} 帧)", fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


def plot_statistics(ax, output_dir: str, traj, odom, images_idx, nodes, lidar_dir):
    """数据统计面板。"""
    ax.axis("off")

    t0 = int(traj[0]["timestamp_ns"])
    t1 = int(traj[-1]["timestamp_ns"])
    duration = (t1 - t0) / 1e9

    tx = [float(r["tx"]) for r in traj]
    ty = [float(r["ty"]) for r in traj]
    path_len = sum(math.sqrt((tx[i]-tx[i-1])**2 + (ty[i]-ty[i-1])**2) for i in range(1, len(tx)))

    lin_v = [math.sqrt(float(r["lin_vel_x"])**2 + float(r["lin_vel_y"])**2) for r in odom]
    avg_v = np.mean(lin_v) if lin_v else 0

    # 点云统计：数 PCD 文件
    pcd_files = _glob.glob(os.path.join(lidar_dir, "*.pcd")) if os.path.isdir(lidar_dir) else []
    lidar_info = f"{len(pcd_files)} 帧" if pcd_files else "N/A"

    # 深度图统计
    depth_dir = os.path.join(output_dir, "depth")
    depth_files = _glob.glob(os.path.join(depth_dir, "*.png")) if os.path.isdir(depth_dir) else []
    depth_info = f"{len(depth_files)} 帧" if depth_files else "N/A"

    stats = [
        ("[Time] 巡检时长", f"{duration:.1f} 秒"),
        ("[Dist] 轨迹长度", f"{path_len:.1f} m"),
        ("[Pts] 轨迹点", f"{len(traj):,}"),
        ("[Odom] 里程计消息", f"{len(odom):,}"),
        ("[Img] RGB图像", f"{len(images_idx):,} 帧"),
        ("[Depth] 深度图", depth_info),
        ("[Node] 记忆节点", f"{len(nodes):,}"),
        ("[LiDAR] 激光雷达", lidar_info),
        ("[Vel] 平均速度", f"{avg_v:.3f} m/s"),
        ("[Res] 图像分辨率", f"{images_idx[0]['width']}x{images_idx[0]['height']}"),
        ("[Robot] 平台", "Deep Robotics Lite3"),
        ("[LiDAR] 激光雷达", "Livox MID-360"),
        ("[Cam] 相机", "Orbbec Gemini 330"),
    ]

    y = 0.95
    for label, value in stats:
        ax.text(0.05, y, f"{label}", fontsize=10, fontweight="bold", va="top",
                transform=ax.transAxes, color="#555555")
        ax.text(0.45, y, f"{value}", fontsize=10, va="top",
                transform=ax.transAxes, color="#111111")
        y -= 0.075

    ax.set_title("数据统计", fontweight="bold")


# ── 主图 ──────────────────────────────────────────────────

def visualize(output_dir: str):
    """生成综合可视化大图。"""
    print(f"加载数据: {output_dir}")

    odom = load_csv(os.path.join(output_dir, "odometry.csv"))
    images_idx = load_csv(os.path.join(output_dir, "images_index.csv"))
    nodes = load_json(os.path.join(output_dir, "memory_nodes.json"))
    lidar_dir = os.path.join(output_dir, "lidar")

    # 轨迹：优先用 map 坐标系合成轨迹，否则 fallback 到 odom 帧
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    map_traj = _compose_map_trajectory(output_dir)
    map_id = _detect_map_id(output_dir, project_dir)
    map_img, map_extent, rooms = None, None, []
    map_label = ""

    if map_traj:
        traj = map_traj
        scenes_dir = os.path.join(project_dir, "datasets", "scenes")
        if map_id:
            map_img, map_extent, rooms = _load_map_background(scenes_dir, map_id)
            map_label = f"地图 {map_id}" if map_img is not None else f"map→base_link (无地图图)"
        else:
            map_label = "map→base_link (无地图绑定)"
        print(f"  轨迹={len(traj)} (map 坐标系)  map_id={map_id or '无'}")
    else:
        traj = load_csv(os.path.join(output_dir, "base_link_pose.csv"))
        print(f"  轨迹={len(traj)} (odom 坐标系, 无法合成 map)")

    print(f"  里程计={len(odom)} 图像={len(images_idx)} 节点={len(nodes)}")

    # ── 创建画布 ──
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(3, 3, figure=fig,
                  width_ratios=[1.2, 1.0, 0.8],
                  height_ratios=[1.0, 0.6, 0.6],
                  hspace=0.35, wspace=0.35)

    # (0,0): 2D 轨迹图（叠加地图）
    ax_traj = fig.add_subplot(gs[0, 0])
    plot_trajectory(ax_traj, traj, nodes, images_idx,
                    map_img=map_img, map_extent=map_extent, rooms=rooms,
                    map_label=map_label)

    # (0,1): 点云俯视
    ax_lidar = fig.add_subplot(gs[0, 1])
    plot_lidar_overhead(ax_lidar, lidar_dir)

    # (0,2): 统计面板
    ax_stats = fig.add_subplot(gs[0, 2])
    plot_statistics(ax_stats, output_dir, traj, odom, images_idx, nodes, lidar_dir)

    # (1, :): 图像缩略图条
    ax_strip = fig.add_subplot(gs[1, :])
    plot_image_strip(ax_strip, output_dir, images_idx, nodes)

    # (2, :): 速度曲线
    ax_vel = fig.add_subplot(gs[2, :])
    plot_velocity(ax_vel, odom)

    # ── 总标题 ──
    fig.suptitle("ScribeMem-Bench 数据集全景 | Deep Robotics Lite3 机器狗巡检记录",
                 fontsize=16, fontweight="bold", y=0.98)

    # ── 保存 ──
    out_path = os.path.join(output_dir, "dataset_overview.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n✓ 可视化已保存: {out_path}")

    return out_path


# ── 内存节点详情图（单独一张） ──

def visualize_memory_nodes(output_dir: str):
    """生成记忆节点详情图。"""
    nodes = load_json(os.path.join(output_dir, "memory_nodes.json"))
    images_idx = load_csv(os.path.join(output_dir, "images_index.csv"))
    img_map = {img["filename"]: img for img in images_idx}
    images_dir = os.path.join(output_dir, "images")

    # 选取最多 20 个节点
    show_nodes = nodes[:20]
    n = len(show_nodes)
    cols = 5
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.8))
    if hasattr(axes, "flatten"):
        axes = axes.flatten()
    else:
        axes = np.array([axes])

    for i, node in enumerate(show_nodes):
        ax = axes[i]
        # 找关联图像
        img_shown = False
        for ref in node.get("image_refs", []):
            fname = os.path.basename(ref)
            img_path = os.path.join(images_dir, fname)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                img_shown = True
                break

        if not img_shown:
            ax.text(0.5, 0.5, "无图像", ha="center", va="center", transform=ax.transAxes,
                    color="gray")

        pos = node["spatial_data"]["position"]
        ax.set_title(f"节点 #{node['node_id']}\n({pos['x']:.1f},{pos['y']:.1f},{pos['z']:.1f})",
                     fontsize=8)
        ax.axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("记忆节点详情 (每个节点的关联图像 + 空间坐标)",
                 fontsize=13, fontweight="bold", y=1.01)

    out_path = os.path.join(output_dir, "memory_nodes_detail.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ 节点详情已保存: {out_path}")

    return out_path


# ── CLI ───────────────────────────────────────────────────

def _discover_sessions(datasets_root: str) -> list[str]:
    """自动发现 datasets/ 下所有包含 base_link_pose.csv 的 session 目录。"""
    sessions = []
    for d in sorted(os.listdir(datasets_root)):
        full = os.path.join(datasets_root, d)
        if os.path.isdir(full) and os.path.exists(os.path.join(full, "base_link_pose.csv")):
            sessions.append(full)
    return sessions


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_root = os.path.join(project_dir, "datasets")

    if len(sys.argv) > 1:
        # 单 session: python3 visualize_dataset.py datasets/session_001
        targets = [sys.argv[1]]
    else:
        # 批量: 自动发现所有 session 目录
        targets = _discover_sessions(default_root)
        if not targets:
            print(f"错误: 在 '{default_root}' 下未找到任何 session 目录。请先运行 assemble_dataset.py")
            sys.exit(1)

    processed = []
    for output_dir in targets:
        if not os.path.exists(os.path.join(output_dir, "base_link_pose.csv")):
            print(f"⚠ 跳过 '{output_dir}'：无提取数据")
            continue
        try:
            p1 = visualize(output_dir)
            p2 = visualize_memory_nodes(output_dir)
            processed.append((output_dir, p1, p2))
        except Exception as e:
            print(f"✗ 处理 '{output_dir}' 失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"✓ 批量完成: {len(processed)}/{len(targets)} 个 session")
    for d, p1, p2 in processed:
        print(f"  {os.path.basename(d)}:")
        print(f"    xdg-open {p1}")
        print(f"    xdg-open {p2}")
