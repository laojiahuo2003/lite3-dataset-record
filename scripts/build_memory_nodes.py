#!/usr/bin/env python3
"""
基于提取的原始数据构建 ScribeMem-Bench 格式的记忆节点。

记忆节点采样策略：
  - 沿轨迹每 N 秒（默认 2s）或每移动 M 米（默认 0.5m）创建一个节点
  - 每个节点关联时间最近的一帧 RGB 图像
  - 每个节点记录该时刻的位姿和激光雷达点云统计

输出:
  output/memory_nodes.json    - 记忆节点列表
  output/dataset_manifest.yaml - 数据集清单
"""

import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _read_csv_with_header(path: str) -> list[dict]:
    """读取 CSV 文件为 dict 列表。"""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def _find_nearest_image(
    target_ts: int,
    image_index: list[dict],
    used_images: set,
) -> Optional[dict]:
    """
    找到与目标时间戳最近的未使用图像帧。

    Args:
        target_ts: 目标时间戳 (ns)
        image_index: 图像索引列表
        used_images: 已使用的图像文件名集合

    Returns:
        最近的图像索引条目，如果没找到返回 None
    """
    best = None
    best_diff = float("inf")
    for img in image_index:
        if img["filename"] in used_images:
            continue
        ts = int(img["timestamp_ns"])
        diff = abs(ts - target_ts)
        if diff < best_diff:
            best_diff = diff
            best = img
    if best:
        used_images.add(best["filename"])
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


def build_memory_nodes(
    output_dir: str,
    sample_interval_s: float = 2.0,
    sample_distance_m: float = 0.5,
) -> list[dict]:
    """
    构建记忆节点。

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
    print(f"已加载: {len(trajectory)} 个轨迹点, {len(odometry)} 条里程计, "
          f"{len(image_index)} 帧RGB, {len(depth_index)} 帧深度")

    # 使用轨迹数据（优先）或里程计数据
    poses = []
    if trajectory:
        for row in trajectory:
            poses.append({
                "ts": int(row["timestamp_ns"]),
                "frame_id": row.get("frame_id", "odom"),
                "tx": float(row["tx"]), "ty": float(row["ty"]), "tz": float(row["tz"]),
                "rx": float(row["rx"]), "ry": float(row["ry"]),
                "rz": float(row["rz"]), "rw": float(row["rw"]),
            })
    else:
        for row in odometry:
            poses.append({
                "ts": int(row["timestamp_ns"]),
                "frame_id": "odom",
                "tx": float(row["pos_x"]), "ty": float(row["pos_y"]), "tz": float(row["pos_z"]),
                "rx": float(row["ori_x"]), "ry": float(row["ori_y"]),
                "rz": float(row["ori_z"]), "rw": float(row["ori_w"]),
            })

    poses.sort(key=lambda p: p["ts"])

    if not poses:
        print("错误: 无有效位姿数据")
        return []

    # 下采样：时间间隔 + 空间间隔
    start_ts = poses[0]["ts"]
    sample_interval_ns = int(sample_interval_s * 1e9)

    nodes = []
    used_images = set()
    used_depths = set()
    last_key_pos = (poses[0]["tx"], poses[0]["ty"], poses[0]["tz"])
    last_key_ts = start_ts

    for pose in poses:
        dt = pose["ts"] - last_key_ts
        curr_pos = (pose["tx"], pose["ty"], pose["tz"])
        dd = _calc_distance(curr_pos, last_key_pos)

        if dt < sample_interval_ns and dd < sample_distance_m:
            continue

        # 找到最近的 RGB 和深度图像
        img = _find_nearest_image(pose["ts"], image_index, used_images)
        depth_img = _find_nearest_image(pose["ts"], depth_index, used_depths) if has_depth else None

        node_id = len(nodes) + 1
        t = pose["ts"]
        rel_ts = t - start_ts

        # 构建图像引用
        image_refs = []
        if img:
            image_refs.append(f"data/images/{img['filename']}")
        if depth_img:
            image_refs.append(f"data/depth/{depth_img['filename']}")

        node = {
            "node_id": node_id,
            "node_type": "LongTerm",
            "summary": (
                f"巡检点 #{node_id}: "
                f"位姿({pose['tx']:.2f}, {pose['ty']:.2f}, {pose['tz']:.2f}), "
                f"相对时间 {rel_ts / 1e9:.1f}s"
            ),
            "timestamp": t,
            "relative_timestamp_ns": rel_ts,
            "spatial_data": {
                "position": {
                    "x": round(pose["tx"], 4),
                    "y": round(pose["ty"], 4),
                    "z": round(pose["tz"], 4),
                },
                "orientation": {
                    "x": round(pose["rx"], 4),
                    "y": round(pose["ry"], 4),
                    "z": round(pose["rz"], 4),
                    "w": round(pose["rw"], 4),
                },
                "frame_id": pose["frame_id"],
            },
            "tags": {
                "scene_type": "unknown",  # 需要后续 VLM 标注
                "objects_present": [],     # 需要后续物体检测
                "action_type": "navigate",
                "success": True,
                "task_type": "inspection",
                "difficulty": "easy",
                "intent": f"机器狗巡检第 {node_id} 个观测点",
            },
            "causal_chain": [node_id - 1] if node_id > 1 else [],
            "weight": 0.8,
            "image_refs": image_refs,
            "access_count": 1,
            "version": 1,
        }

        nodes.append(node)
        last_key_pos = curr_pos
        last_key_ts = pose["ts"]

    print(f"✓ 生成 {len(nodes)} 个记忆节点 (采样间隔: {sample_interval_s}s / {sample_distance_m}m)")

    # 写入 JSON
    nodes_path = os.path.join(output_dir, "memory_nodes.json")
    with open(nodes_path, "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, indent=2)
    print(f"  记忆节点: → {nodes_path}")

    return nodes


def build_dataset_manifest(
    output_dir: str,
    bag_dir: str,
    num_nodes: int,
    num_images: int,
) -> dict:
    """生成数据集清单 (dataset_manifest.yaml)。"""
    manifest = {
        "benchmark": "ScribeMem-Bench v1.0",
        "source": {
            "bag_dir": bag_dir,
            "platform": "Deep Robotics Lite3 (quadruped robot dog)",
            "sensors": [
                "Orbbec Gemini 330 RGB-D camera",
                "Livox MID-360 LiDAR",
                "Built-in IMU + odometry",
            ],
        },
        "extraction_date": datetime.now(timezone.utc).isoformat(),
        "statistics": {
            "num_memory_nodes": num_nodes,
            "num_images": num_images,
            "num_topics": 5,
            "topics": [
                "/odom",
                "/tf",
                "/tf_static",
                "/camera/color/image_raw",
                "/scanner/cloud",
            ],
        },
        "data_format": {
            "memory_nodes": "memory_nodes.json",
            "odometry": "odometry.csv",
            "trajectory": "base_link_pose.csv",
            "tf_all": "tf_trajectory.csv",
            "images": "images/",
            "images_index": "images_index.csv",
            "lidar": "lidar/",
            "lidar_points": "lidar_points.csv",
        },
    }

    manifest_path = os.path.join(output_dir, "dataset_manifest.yaml")
    import yaml
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.dump(manifest, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"✓ 数据集清单: → {manifest_path}")

    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="构建 ScribeMem-Bench 记忆节点")
    parser.add_argument("--output-dir", default="datasets/session_001", help="提取数据的输出目录")
    parser.add_argument("--bag-dir", default="raw/session_001", help="rosbag2 目录")
    parser.add_argument("--sample-interval", type=float, default=2.0, help="时间采样间隔(秒)")
    parser.add_argument("--sample-distance", type=float, default=0.5, help="空间采样间隔(米)")
    args = parser.parse_args()

    import yaml

    nodes = build_memory_nodes(
        args.output_dir,
        sample_interval_s=args.sample_interval,
        sample_distance_m=args.sample_distance,
    )

    image_index = _read_csv_with_header(os.path.join(args.output_dir, "images_index.csv"))

    build_dataset_manifest(
        args.output_dir,
        args.bag_dir,
        len(nodes),
        len(image_index),
    )
