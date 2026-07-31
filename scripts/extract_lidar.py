#!/usr/bin/env python3
"""
从 rosbag2 中提取 /scanner/cloud (PointCloud2) 话题数据。

输出:
  - PCD 文件 (每帧一个)  → output/lidar/cloud_000000.pcd
"""

import os
import struct
import sys

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2


def extract_lidar(bag_dir: str, output_dir: str, save_pcd: bool = True) -> int:
    """
    提取 LiDAR 点云数据。

    Args:
        bag_dir: rosbag2 目录路径
        output_dir: 输出目录
        save_pcd: 是否保存 PCD 文件

    Returns:
        提取的点云帧数
    """
    storage_opts = StorageOptions(uri=bag_dir, storage_id="sqlite3")
    converter_opts = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = SequentialReader()
    reader.open(storage_opts, converter_opts)

    # 检查话题是否存在
    topic_names = {t.name for t in reader.get_all_topics_and_types()}
    if "/scanner/cloud" not in topic_names:
        print("  ⚠ 未找到 /scanner/cloud 话题，跳过")
        return 0

    lidar_dir = os.path.join(output_dir, "lidar")
    os.makedirs(lidar_dir, exist_ok=True)

    count = 0
    total_points = 0

    while reader.has_next():
        topic_name, msg_data, timestamp = reader.read_next()
        if topic_name != "/scanner/cloud":
            continue

        cloud: PointCloud2 = deserialize_message(msg_data, PointCloud2)

        # 查找各字段的偏移量和类型
        field_offsets = {f.name: (f.offset, f.datatype) for f in cloud.fields}

        def _read_field(row, name, default=0.0):
            if name not in field_offsets:
                return default
            off, dtype = field_offsets[name]
            if dtype == 7:    # FLOAT32
                return struct.unpack_from("f", row, off)[0]
            elif dtype == 8:  # FLOAT64
                return struct.unpack_from("d", row, off)[0]
            elif dtype == 2:  # UINT8
                return struct.unpack_from("B", row, off)[0]
            return struct.unpack_from("f", row, off)[0]

        points = []
        for i in range(cloud.height * cloud.width):
            offset = i * cloud.point_step
            row = cloud.data[offset : offset + cloud.point_step]

            x = _read_field(row, "x")
            y = _read_field(row, "y")
            z = _read_field(row, "z")
            intensity = _read_field(row, "intensity")
            line = _read_field(row, "line")
            pt_ts = _read_field(row, "timestamp")

            points.append((x, y, z, intensity, line, pt_ts))

        total_points += len(points)

        if save_pcd:
            pcd_path = os.path.join(lidar_dir, f"cloud_{count:06d}.pcd")
            _write_pcd(pcd_path, points)

        count += 1
        if count % 100 == 0:
            print(f"  已提取 {count} 帧点云 ({total_points} 点)...")

    print(f"✓ 点云数据: {count} 帧, {total_points} 点 → {lidar_dir}/")
    return count


def _write_pcd(filepath: str, points: list) -> None:
    """写 ASCII PCD 文件（完整7字段）。"""
    with open(filepath, "w") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z intensity line timestamp\n")
        f.write("SIZE 4 4 4 4 1 8\n")
        f.write("TYPE F F F F U F\n")
        f.write("COUNT 1 1 1 1 1 1\n")
        f.write(f"WIDTH {len(points)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(points)}\n")
        f.write("DATA ascii\n")
        for x, y, z, intensity, line, pt_ts in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {intensity:.6f} {int(line)} {pt_ts:.9f}\n")


if __name__ == "__main__":
    bag_dir = sys.argv[1] if len(sys.argv) > 1 else "raw/session_001"
    output = sys.argv[2] if len(sys.argv) > 2 else "datasets/session_001"
    extract_lidar(bag_dir, output)
