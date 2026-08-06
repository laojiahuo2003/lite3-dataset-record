#!/usr/bin/env python3
"""
从 rosbag2 中提取 /tf 和 /tf_static 话题数据。

输出:
  - output/tf_trajectory.csv  (每个 transform 一行)
  - output/base_link_pose.csv  (仅提取 base_link 在 odom/map 下的位姿，即机器狗轨迹)
"""

import csv
import math
import os
import sys

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from tf2_msgs.msg import TFMessage


def _quat_to_euler(rx: float, ry: float, rz: float, rw: float) -> tuple[float, float, float]:
    """Convert quaternion to roll, pitch, yaw (intrinsic ZYX / Tait-Bryan)."""
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (rw * rx + ry * rz)
    cosr_cosp = 1.0 - 2.0 * (rx * rx + ry * ry)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (rw * ry - rz * rx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (rw * rz + rx * ry)
    cosy_cosp = 1.0 - 2.0 * (ry * ry + rz * rz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def extract_tf(bag_dir: str, output_dir: str) -> dict:
    """
    提取所有 TF 变换数据。

    Returns:
        {"/tf": count, "/tf_static": count}
    """
    storage_opts = StorageOptions(uri=bag_dir, storage_id="sqlite3")
    converter_opts = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = SequentialReader()
    reader.open(storage_opts, converter_opts)

    os.makedirs(output_dir, exist_ok=True)

    tf_csv = os.path.join(output_dir, "tf_trajectory.csv")
    trajectory_csv = os.path.join(output_dir, "base_link_pose.csv")

    counts = {"/tf": 0, "/tf_static": 0}

    with open(tf_csv, "w", newline="") as f_all, \
         open(trajectory_csv, "w", newline="") as f_traj:

        writer_all = csv.writer(f_all)
        writer_all.writerow([
            "t",
            "parent_frame", "child_frame",
            "tx", "ty", "tz", "qx", "qy", "qz", "qw",
        ])

        writer_traj = csv.writer(f_traj)
        writer_traj.writerow(["t", "x", "y", "z", "roll", "pitch", "yaw"])

        while reader.has_next():
            topic_name, msg_data, timestamp = reader.read_next()
            if topic_name not in ("/tf", "/tf_static"):
                continue

            tf_msg: TFMessage = deserialize_message(msg_data, TFMessage)

            for transform in tf_msg.transforms:
                writer_all.writerow([
                    timestamp,
                    transform.header.frame_id, transform.child_frame_id,
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w,
                ])

                # 提取 base_link 轨迹（欧拉角格式）
                if transform.child_frame_id == "base_link":
                    roll, pitch, yaw = _quat_to_euler(
                        transform.transform.rotation.x,
                        transform.transform.rotation.y,
                        transform.transform.rotation.z,
                        transform.transform.rotation.w,
                    )
                    writer_traj.writerow([
                        timestamp,
                        transform.transform.translation.x,
                        transform.transform.translation.y,
                        transform.transform.translation.z,
                        roll, pitch, yaw,
                    ])

            counts[topic_name] += 1

            total = sum(counts.values())
            if total % 5000 == 0:
                print(f"  已处理 {total} 条 TF 消息...")

    print(f"✓ TF 全部变换: {counts['/tf'] + counts['/tf_static']} 条 → {tf_csv}")
    print(f"  base_link 轨迹: → {trajectory_csv}")
    return counts


if __name__ == "__main__":
    bag_dir = sys.argv[1] if len(sys.argv) > 1 else "raw/session_001"
    output = sys.argv[2] if len(sys.argv) > 2 else "datasets/session_001"
    extract_tf(bag_dir, output)
