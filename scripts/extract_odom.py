#!/usr/bin/env python3
"""
从 rosbag2 中提取 /odom 话题数据，输出 CSV 文件。

输出列：
  timestamp_ns, pos_x, pos_y, pos_z,
  ori_x, ori_y, ori_z, ori_w,
  lin_vel_x, lin_vel_y, lin_vel_z,
  ang_vel_x, ang_vel_y, ang_vel_z,
  child_frame_id
"""

import csv
import os
import sys
from pathlib import Path

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry


def extract_odometry(bag_dir: str, output_csv: str) -> int:
    """
    提取里程计数据到 CSV。

    Returns:
        提取的消息数量
    """
    # 配置 rosbag2 reader
    storage_opts = StorageOptions(uri=bag_dir, storage_id="sqlite3")
    converter_opts = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = SequentialReader()
    reader.open(storage_opts, converter_opts)

    # 检查 /odom 话题是否存在
    topic_names = {t.name for t in reader.get_all_topics_and_types()}
    if "/odom" not in topic_names:
        print("错误: 未找到 /odom 话题", file=sys.stderr)
        return 0

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    count = 0

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_ns",
            "pos_x", "pos_y", "pos_z",
            "ori_x", "ori_y", "ori_z", "ori_w",
            "lin_vel_x", "lin_vel_y", "lin_vel_z",
            "ang_vel_x", "ang_vel_y", "ang_vel_z",
            "child_frame_id",
        ])

        while reader.has_next():
            topic_name, msg_data, timestamp = reader.read_next()
            if topic_name != "/odom":
                continue

            odom: Odometry = deserialize_message(msg_data, Odometry)

            writer.writerow([
                timestamp,
                odom.pose.pose.position.x,
                odom.pose.pose.position.y,
                odom.pose.pose.position.z,
                odom.pose.pose.orientation.x,
                odom.pose.pose.orientation.y,
                odom.pose.pose.orientation.z,
                odom.pose.pose.orientation.w,
                odom.twist.twist.linear.x,
                odom.twist.twist.linear.y,
                odom.twist.twist.linear.z,
                odom.twist.twist.angular.x,
                odom.twist.twist.angular.y,
                odom.twist.twist.angular.z,
                odom.child_frame_id,
            ])
            count += 1

            if count % 5000 == 0:
                print(f"  已提取 {count} 条里程计数据...")

    print(f"✓ 里程计数据: {count} 条 → {output_csv}")
    return count


if __name__ == "__main__":
    bag_dir = sys.argv[1] if len(sys.argv) > 1 else "raw/session_001"
    output = sys.argv[2] if len(sys.argv) > 2 else "datasets/session_001/odometry.csv"
    extract_odometry(bag_dir, output)
