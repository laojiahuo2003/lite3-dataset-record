#!/usr/bin/env python3
"""
从 rosbag2 中提取 /camera/color/image_raw 话题数据，保存为 PNG 文件。

每帧的图像索引和时间戳记录在 images_index.csv 中。

输出:
  output/images/frame_000000.png
  output/images/frame_000001.png
  ...
  output/images_index.csv
"""

import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image


def image_to_ndarray(msg: Image) -> np.ndarray:
    """
    将 sensor_msgs/Image 转为 numpy 数组 (H, W, C) uint8。

    支持: rgb8, bgr8, bgra8, mono8, mono16
    """
    dtype_map = {
        "mono8": (np.uint8, 1),
        "8UC1": (np.uint8, 1),
        "mono16": (np.uint16, 1),
        "16UC1": (np.uint16, 1),
        "bgr8": (np.uint8, 3),
        "8UC3": (np.uint8, 3),
        "rgb8": (np.uint8, 3),
        "bgra8": (np.uint8, 4),
        "8UC4": (np.uint8, 4),
        "rgba8": (np.uint8, 4),
    }
    dtype, channels = dtype_map.get(msg.encoding, (np.uint8, 3))
    arr = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, channels)
    return arr


def extract_images(bag_dir: str, output_dir: str) -> int:
    """
    提取相机图像到 PNG 文件。

    Returns:
        提取的图像数量
    """
    storage_opts = StorageOptions(uri=bag_dir, storage_id="sqlite3")
    converter_opts = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = SequentialReader()
    reader.open(storage_opts, converter_opts)

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    index_path = os.path.join(output_dir, "images_index.csv")
    count = 0

    with open(index_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "ts", "file_path",
                         "cam_x", "cam_y", "cam_z",
                         "cam_qx", "cam_qy", "cam_qz", "cam_qw"])

        while reader.has_next():
            topic_name, msg_data, timestamp = reader.read_next()
            if topic_name != "/camera/color/image_raw":
                continue

            img_msg: Image = deserialize_message(msg_data, Image)

            filename = f"frame_{count:06d}.png"
            filepath = os.path.join(images_dir, filename)

            arr = image_to_ndarray(img_msg)

            # bgr8 → RGB for proper PNG saving
            if img_msg.encoding in ("bgr8", "8UC3"):
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            elif img_msg.encoding in ("bgra8", "8UC4"):
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)

            cv2.imwrite(filepath, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if arr.ndim == 3 and arr.shape[2] == 3 else arr)

            writer.writerow([
                count, timestamp, f"images/{filename}",
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,  # camera pose: filled by assemble later
            ])
            count += 1

            if count % 200 == 0:
                print(f"  已提取 {count} 帧图像...")

    print(f"✓ 图像数据: {count} 帧 → {images_dir}/")
    print(f"  索引文件: {index_path}")
    return count


if __name__ == "__main__":
    bag_dir = sys.argv[1] if len(sys.argv) > 1 else "raw/session_001"
    output = sys.argv[2] if len(sys.argv) > 2 else "datasets/session_001"
    extract_images(bag_dir, output)
