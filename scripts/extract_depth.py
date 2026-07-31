#!/usr/bin/env python3
"""
从 rosbag2 中提取深度图话题数据，保存为 16-bit PNG。

自动检测常见的深度话题名:
  - /camera/aligned_depth_to_color/image_raw   (已对齐到彩色, 优先)
  - /camera/depth/image_rect_raw               (深度原生)
  - /camera/depth/image_raw                    (深度原始)

输出:
  output/depth/frame_000000.png
  output/depth_index.csv
"""

import csv
import os
import sys

import cv2
import numpy as np
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image


# 候选深度话题名（按优先级排序）
_DEPTH_TOPIC_CANDIDATES = [
    # Orbbec Gemini 330 / RealSense — 对齐到彩色帧（最优）
    "/camera/aligned_depth_to_color/image_raw",
    # Orbbec Gemini 330 — 深度原生帧
    "/camera/depth/image_raw",
    # RealSense D455 常用话题
    "/camera/depth/image_rect_raw",
    # Orbbec Gemini 330 — 未对齐深度（备选）
    "/camera/depth/image_unaligned",
]


def _find_depth_topic(reader: SequentialReader) -> str | None:
    """自动检测 bag 中的深度话题。"""
    topics = {t.name for t in reader.get_all_topics_and_types()}
    for candidate in _DEPTH_TOPIC_CANDIDATES:
        if candidate in topics:
            return candidate
    # 模糊匹配
    for t in topics:
        if "depth" in t.lower() and "image" in t.lower():
            return t
    return None


def _depth_to_mm(arr: np.ndarray, encoding: str) -> np.ndarray:
    """
    将深度图统一转换为 uint16 毫米值。
    输入可能是 mono16(单位mm) 或 32FC1(单位m)。
    """
    if encoding in ("mono16", "16UC1"):
        return arr.astype(np.uint16)
    elif encoding in ("32FC1",):
        return (arr * 1000.0).clip(0, 65535).astype(np.uint16)
    elif encoding in ("mono8", "8UC1"):
        return arr.astype(np.uint16)
    else:
        # 未知编码，尝试保持原样
        print(f"    警告: 未知深度编码 '{encoding}'，保持原样")
        return arr


def extract_depth(bag_dir: str, output_dir: str) -> int:
    """
    提取深度图到 PNG 文件。

    Returns:
        提取的深度帧数 (0 表示无深度话题)
    """
    storage_opts = StorageOptions(uri=bag_dir, storage_id="sqlite3")
    converter_opts = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = SequentialReader()
    reader.open(storage_opts, converter_opts)

    depth_topic = _find_depth_topic(reader)
    if depth_topic is None:
        print("  ⚠ 未找到深度话题，跳过 (bag中无depth话题)")
        return 0

    print(f"  检测到深度话题: {depth_topic}")

    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)

    index_path = os.path.join(output_dir, "depth_index.csv")
    count = 0

    with open(index_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "timestamp_ns", "filename", "width", "height", "encoding"])

        while reader.has_next():
            topic_name, msg_data, timestamp = reader.read_next()
            if topic_name != depth_topic:
                continue

            depth_msg: Image = deserialize_message(msg_data, Image)

            # 解析深度数据
            if depth_msg.encoding in ("mono16", "16UC1"):
                arr = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(
                    depth_msg.height, depth_msg.width
                )
            elif depth_msg.encoding in ("32FC1",):
                arr = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(
                    depth_msg.height, depth_msg.width
                )
            elif depth_msg.encoding in ("mono8", "8UC1"):
                arr = np.frombuffer(depth_msg.data, dtype=np.uint8).reshape(
                    depth_msg.height, depth_msg.width
                )
            else:
                # 尝试 uint16
                arr = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(
                    depth_msg.height, depth_msg.width
                )

            arr_mm = _depth_to_mm(arr, depth_msg.encoding)

            filename = f"frame_{count:06d}.png"
            filepath = os.path.join(depth_dir, filename)
            cv2.imwrite(filepath, arr_mm)

            writer.writerow([
                count, timestamp, filename,
                depth_msg.width, depth_msg.height, depth_msg.encoding,
            ])
            count += 1

            if count % 200 == 0:
                print(f"  已提取 {count} 帧深度图...")

    if count > 0:
        print(f"✓ 深度图数据: {count} 帧 → {depth_dir}/")
        print(f"  索引文件: {index_path}")
    return count


if __name__ == "__main__":
    bag_dir = sys.argv[1] if len(sys.argv) > 1 else "raw/session_001"
    output = sys.argv[2] if len(sys.argv) > 2 else "datasets/session_001"
    extract_depth(bag_dir, output)
