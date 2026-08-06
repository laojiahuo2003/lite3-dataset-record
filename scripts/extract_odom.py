#!/usr/bin/env python3
"""
从 rosbag2 中提取 /odom 话题的速度数据，输出 CSV 文件。

输出列（附录 G 规范）：
  t, v_linear_x, v_linear_y, v_angular_z
"""

import csv
import os
import sys

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry


def extract_odometry(bag_dir: str, output_csv: str) -> int:
    """提取里程计速度到 CSV。"""
    storage_opts = StorageOptions(uri=bag_dir, storage_id="sqlite3")
    converter_opts = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = SequentialReader()
    reader.open(storage_opts, converter_opts)

    topic_names = {t.name for t in reader.get_all_topics_and_types()}
    if "/odom" not in topic_names:
        print("错误: 未找到 /odom 话题", file=sys.stderr)
        return 0

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    count = 0

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "v_linear_x", "v_linear_y", "v_angular_z"])

        while reader.has_next():
            topic_name, msg_data, timestamp = reader.read_next()
            if topic_name != "/odom":
                continue

            odom: Odometry = deserialize_message(msg_data, Odometry)

            writer.writerow([
                timestamp,
                odom.twist.twist.linear.x,
                odom.twist.twist.linear.y,
                odom.twist.twist.angular.z,
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
