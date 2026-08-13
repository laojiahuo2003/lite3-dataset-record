#!/usr/bin/env python3
"""
基于提取的原始数据构建 ScribeMem-Bench 格式的记忆节点（附录 C 格式）。

记忆节点采样策略：
  - 沿轨迹每 N 秒（默认 2s）或每移动 M 米（默认 0.5m）创建一个节点
  - 每个节点关联时间最近的一帧 RGB 图像 + 深度图
  - 每个节点记录该时刻的位姿和相机参数

输出:
  output/memory_nodes.json    - 记忆节点列表（附录 C 格式）
"""

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional


# ── 节点类型映射 ──────────────────────────────────────────────────────────

def _node_type_from_tags(tags: dict) -> str:
    """从 tags 推断节点类型，默认 LongTerm。"""
    action = tags.get("action_type", "navigate")
    if action == "observe":
        return "LongTerm"
    elif action == "navigate":
        return "ShortTerm"
    return "LongTerm"


# ── 四元数转欧拉角 ────────────────────────────────────────────────────────

def _quat_to_euler(rx: float, ry: float, rz: float, rw: float):
    """Convert quaternion to roll, pitch, yaw."""
    sinr_cosp = 2.0 * (rw * rx + ry * rz)
    cosr_cosp = 1.0 - 2.0 * (rx * rx + ry * ry)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (rw * ry - rz * rx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (rw * rz + rx * ry)
    cosy_cosp = 1.0 - 2.0 * (ry * ry + rz * rz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


# ── CSV 工具 ──────────────────────────────────────────────────────────────

def _read_csv_with_header(path: str) -> list[dict]:
    """读取 CSV 文件为 dict 列表。"""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def _find_nearest_row(
    target_ts: int,
    rows: list[dict],
    used: set,
    ts_col: str = "ts",
    key_col: str = "file_path",
) -> Optional[dict]:
    """找到与目标时间戳最近的未使用行。"""
    best = None
    best_diff = float("inf")
    for row in rows:
        key = row.get(key_col, "")
        if key in used:
            continue
        diff = abs(int(float(row[ts_col])) - target_ts)
        if diff < best_diff:
            best_diff = diff
            best = row
    if best:
        used.add(best.get(key_col, ""))
    return best


def _find_nearest_pose(
    target_ts: int,
    poses: list[dict],
) -> Optional[dict]:
    """找到与目标时间戳最近的里程计位姿。"""
    best = None
    best_diff = float("inf")
    for p in poses:
        diff = abs(int(p["timestamp_ns"]) - target_ts)
        if diff < best_diff:
            best_diff = diff
            best = p
    return best


def _calc_distance(
    pos1: tuple[float, float, float],
    pos2: tuple[float, float, float],
) -> float:
    """计算两点欧式距离。"""
    return math.sqrt(
        (pos1[0] - pos2[0]) ** 2
        + (pos1[1] - pos2[1]) ** 2
        + (pos1[2] - pos2[2]) ** 2
    )


# ── 主构建函数 ────────────────────────────────────────────────────────────

def build_memory_nodes(
    output_dir: str,
    sample_interval_s: float = 2.0,
    sample_distance_m: float = 0.5,
) -> list[dict]:
    """
    构建附录 C 格式的记忆节点。

    Args:
        output_dir: 已提取数据的输出目录
        sample_interval_s: 时间采样间隔（秒）
        sample_distance_m: 空间采样间隔（米）

    Returns:
        记忆节点列表
    """
    # 加载已提取的数据
    trajectory = _read_csv_with_header(os.path.join(output_dir, "base_link_pose.csv"))
    odometry = _read_csv_with_header(os.path.join(output_dir, "odometry.csv"))
    image_index = _read_csv_with_header(os.path.join(output_dir, "images_index.csv"))
    depth_index = _read_csv_with_header(os.path.join(output_dir, "depth_index.csv"))

    has_depth = len(depth_index) > 0
    has_odom = len(odometry) > 0

    print(f"已加载: {len(trajectory)} 个轨迹点, {len(odometry)} 条里程计, "
          f"{len(image_index)} 帧RGB, {len(depth_index)} 帧深度")

    # 提取位姿序列（优先用 base_link_pose.csv，回退到 odometry.csv）
    poses = []
    if trajectory:
        for row in trajectory:
            poses.append({
                "ts": int(float(row["t"])),
                "tx": float(row["x"]), "ty": float(row["y"]), "tz": float(row["z"]),
                "roll": float(row["roll"]), "pitch": float(row["pitch"]), "yaw": float(row["yaw"]),
            })
    elif has_odom:
        for row in odometry:
            rx = float(row["ori_x"]); ry = float(row["ori_y"])
            rz = float(row["ori_z"]); rw = float(row["ori_w"])
            roll, pitch, yaw = _quat_to_euler(rx, ry, rz, rw)
            poses.append({
                "ts": int(row["timestamp_ns"]),
                "tx": float(row["pos_x"]), "ty": float(row["pos_y"]), "tz": float(row["pos_z"]),
                "roll": roll, "pitch": pitch, "yaw": yaw,
            })

    poses.sort(key=lambda p: p["ts"])

    if not poses:
        print("错误: 无有效位姿数据")
        return []

    # 加载相机标定（如果有）
    calib_path = os.path.join(output_dir, "camera_calib.yaml")
    calib = {}
    if os.path.exists(calib_path):
        try:
            import yaml
            with open(calib_path, "r") as f:
                calib = yaml.safe_load(f) or {}
        except Exception:
            pass

    rgb_calib = calib.get("rgb", {})
    depth_calib = calib.get("depth", {})
    extrinsics = calib.get("extrinsics", {})

    # 默认相机内参
    default_fx = float(rgb_calib.get("fx", 525.0))
    default_fy = float(rgb_calib.get("fy", 525.0))
    default_cx = float(rgb_calib.get("cx", 319.5))
    default_cy = float(rgb_calib.get("cy", 239.5))
    default_width = int(rgb_calib.get("width", 640))
    default_height = int(rgb_calib.get("height", 480))
    default_distortion = [float(x) for x in rgb_calib.get("distortion", [0.0] * 5)]
    default_depth_scale = float(depth_calib.get("depth_scale", 0.001))

    # 下采样：时间间隔 + 空间间隔
    start_ts = poses[0]["ts"]
    sample_interval_ns = int(sample_interval_s * 1e9)

    nodes = []
    used_images: set = set()
    used_depths: set = set()
    last_key_pos = (poses[0]["tx"], poses[0]["ty"], poses[0]["tz"])
    last_key_ts = start_ts

    for pose in poses:
        dt = pose["ts"] - last_key_ts
        curr_pos = (pose["tx"], pose["ty"], pose["tz"])
        dd = _calc_distance(curr_pos, last_key_pos)

        if dt < sample_interval_ns and dd < sample_distance_m:
            continue

        # 找到最近的 RGB 和深度图像
        img_row = _find_nearest_row(pose["ts"], image_index, used_images)
        depth_row = _find_nearest_row(pose["ts"], depth_index, used_depths) if has_depth else None

        # 查找里程计位姿（用于 camera_pose）
        cam_pose = {
            "x": pose["tx"], "y": pose["ty"], "z": pose["tz"],
            "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
        }
        if has_odom:
            odom_pose = _find_nearest_pose(pose["ts"], odometry)
            if odom_pose:
                cam_pose = {
                    "x": float(odom_pose["pos_x"]),
                    "y": float(odom_pose["pos_y"]),
                    "z": float(odom_pose["pos_z"]),
                    "qx": float(odom_pose["ori_x"]),
                    "qy": float(odom_pose["ori_y"]),
                    "qz": float(odom_pose["ori_z"]),
                    "qw": float(odom_pose["ori_w"]),
                }

        node_id = len(nodes) + 1
        ts = pose["ts"]
        rel_ts = ts - start_ts

        # ── 构建图像引用 ────────────────────────────────────────────
        image_refs = []
        depth_refs = []
        frame_ts_list = []

        if img_row:
            image_refs.append(img_row["file_path"])
            frame_ts_list.append(int(float(img_row["ts"])))

        if depth_row:
            depth_refs.append(depth_row["file_path"])
            if not frame_ts_list:
                frame_ts_list.append(int(float(depth_row["ts"])))

        # ── 构建附录 C 格式节点 ──────────────────────────────────────
        summary = (
            f"巡检点 #{node_id}: "
            f"位姿({pose['tx']:.2f}, {pose['ty']:.2f}, {pose['tz']:.2f}), "
            f"相对时间 {rel_ts / 1e9:.1f}s"
        )

        tags = {
            "scene_type": "",
            "objects_present": [],
            "action_type": "navigate",
            "success": True,
            "task_type": "inspection",
            "difficulty": "easy",
            "intent_zh": f"机器狗巡检第 {node_id} 个观测点",
            "intent_en": f"Robot dog patrol observation point #{node_id}",
        }

        camera_params = {
            "fx": default_fx,
            "fy": default_fy,
            "cx": default_cx,
            "cy": default_cy,
            "width": default_width,
            "height": default_height,
            "distortion": default_distortion,
            "camera_pose": cam_pose,
            "depth_scale": default_depth_scale,
            "camera_type": "rgb",
        }

        node = {
            "node_id": node_id,
            "summary_zh": summary,
            "summary_en": summary,
            "timestamp": ts,
            "time_range": {
                "start_ts": ts,
                "end_ts": ts + int(0.5 * 1e9),
            },
            "node_type": _node_type_from_tags(tags),
            "spatial_data": {
                "objects": [],
                "origin": "world",
            },
            "camera_params": camera_params,
            "tags": tags,
            "causal_chain": [node_id - 1] if node_id > 1 else [],
            "weight": 0.8,
            "image_refs": image_refs,
            "depth_refs": depth_refs,
            "frame_ts": frame_ts_list,
            "video_clip_refs": [],
            "confidence_flags": {
                "object_detection": 0.5,
                "tag_classification": 0.5,
            },
            "access_count": 1,
            "version": 1,
        }

        nodes.append(node)
        last_key_pos = curr_pos
        last_key_ts = ts

    print(f"✓ 生成 {len(nodes)} 个记忆节点 (采样间隔: {sample_interval_s}s / {sample_distance_m}m)")

    # 写入 JSON
    nodes_path = os.path.join(output_dir, "memory_nodes.json")
    with open(nodes_path, "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, indent=2)
    print(f"  记忆节点: → {nodes_path}")

    return nodes


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="构建 ScribeMem-Bench 记忆节点（附录 C）")
    parser.add_argument("--output-dir", default="datasets/scenes1/session_001", help="提取数据的输出目录")
    parser.add_argument("--bag-dir", default="raw/session_001", help="rosbag2 目录（保留兼容，不再使用）")
    parser.add_argument("--sample-interval", type=float, default=2.0, help="时间采样间隔(秒)")
    parser.add_argument("--sample-distance", type=float, default=0.5, help="空间采样间隔(米)")
    args = parser.parse_args()

    build_memory_nodes(
        args.output_dir,
        sample_interval_s=args.sample_interval,
        sample_distance_m=args.sample_distance,
    )
