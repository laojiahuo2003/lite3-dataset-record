#!/usr/bin/env python3
"""
从 rosbag2 提取相机内参，生成 camera_calib.yaml。

优先级:
  1. bag 中的 /camera/color/camera_info（Orbbec 出厂标定，首选，离线可复现）
  2. 兜底: templates/camera_calib.yaml（实测的出厂标定值，供旧 bag / 无 camera_info 的 session 使用）

同时估算实际采集帧率（从 camera_info 消息时间戳中位数得出）。

用法:
  python3 scripts/extract_calib.py raw/session_001 datasets/scenes1/session_001
  python3 scripts/extract_calib.py <bag_dir> <output_dir>
"""

import argparse
import os
import sys
from datetime import datetime, timezone

import yaml
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs.msg import CameraInfo

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

ZERO5 = [0.0, 0.0, 0.0, 0.0, 0.0]


def _distortion5(d) -> list:
    """取 OpenCV plumb_bob 前 5 个畸变系数 [k1,k2,p1,p2,k3]（丢弃尾部补零）。"""
    d = list(d or [])
    if len(d) >= 5:
        return d[:5]
    return d + ZERO5[len(d):]


def _read_template() -> dict:
    with open(os.path.join(PROJECT_DIR, "templates", "camera_calib.yaml")) as f:
        return yaml.safe_load(f) or {}


def _extract_from_bag(bag_dir: str):
    """
    从 bag 读取首个 color/depth camera_info 及真实帧率。

    Returns:
        (color_calib: dict, depth_calib: dict, calib_date: str, frame_rate: float)
        未找到任何 camera_info 时 color_calib 为 None。
    """
    storage_opts = StorageOptions(uri=bag_dir, storage_id="sqlite3")
    converter_opts = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = SequentialReader()
    reader.open(storage_opts, converter_opts)

    # 老 bag 未录 camera_info 时直接回退模板，避免全量扫描
    topics = {t.name for t in reader.get_all_topics_and_types()}
    if "/camera/color/camera_info" not in topics and "/camera/depth/camera_info" not in topics:
        return None, None, "", 15.0

    color_info: CameraInfo | None = None
    depth_info: CameraInfo | None = None
    bag_ts = []  # 用于帧率估算

    while reader.has_next():
        topic_name, msg_data, timestamp = reader.read_next()
        if topic_name == "/camera/color/camera_info":
            if color_info is None:
                color_info = deserialize_message(msg_data, CameraInfo)
            bag_ts.append(timestamp)
        elif topic_name == "/camera/depth/camera_info" and depth_info is None:
            depth_info = deserialize_message(msg_data, CameraInfo)

        # 拿到彩色内参 + 足够的时间戳即可收工（深度内参缺失时由彩色镜像补齐）
        if color_info is not None and len(bag_ts) >= 20 and depth_info is not None:
            break

    if color_info is None:
        return None, None, "", 15.0

    # 帧率：取相邻时间戳间隔的中位数
    frame_rate = 15.0
    if len(bag_ts) >= 2:
        dt = sorted((b - a) for a, b in zip(bag_ts, bag_ts[1:]))
        dt = [d for d in dt if d > 0]
        if dt:
            med = sorted(dt)[len(dt) // 2]
            if med > 0:
                frame_rate = round(1.0 / (med / 1e9), 1)

    calib_date = datetime.fromtimestamp(color_info.header.stamp.sec, tz=timezone.utc).strftime("%Y-%m-%d")

    def _from_cam_info(ci: CameraInfo) -> dict:
        k = ci.k
        return {
            "fx": float(k[0]),
            "fy": float(k[4]),
            "cx": float(k[2]),
            "cy": float(k[5]),
            "width": int(ci.width),
            "height": int(ci.height),
            "distortion": _distortion5(ci.d),
        }

    color_calib = _from_cam_info(color_info)
    if depth_info is not None:
        depth_calib = _from_cam_info(depth_info)
    else:
        # 深度已注册到彩色，镜像彩色内参
        depth_calib = dict(color_calib)

    return color_calib, depth_calib, calib_date, frame_rate


def extract_calib(bag_dir: str, output_dir: str) -> int:
    """提取/生成 camera_calib.yaml。返回 0 成功。"""
    calib = _read_template()
    session_id = os.path.basename(output_dir.rstrip("/"))

    color_calib, depth_calib, calib_date, frame_rate = _extract_from_bag(bag_dir)
    source = "bag camera_info"
    if color_calib is None:
        source = "templates (bag 无 camera_info)"
        calib_date = ""
        frame_rate = float(calib.get("rgb", {}).get("frame_rate", 15.0))

    calib["session_id"] = session_id
    calib["calib_date"] = calib_date
    calib["rgb"].update(color_calib or {})
    calib["rgb"]["frame_rate"] = frame_rate
    if depth_calib is not None:
        calib["depth"].update(depth_calib)
    calib["depth"].setdefault("depth_scale", 0.001)
    calib["depth"]["aligned_to"] = "rgb"

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "camera_calib.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(calib, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    rgb = calib["rgb"]
    print(f"✓ 相机标定 → {out_path}")
    print(f"  来源: {source}")
    print(f"  rgb: fx={rgb['fx']:.3f} fy={rgb['fy']:.3f} cx={rgb['cx']:.3f} cy={rgb['cy']:.3f} "
          f"{rgb['width']}x{rgb['height']} @{frame_rate}fps 失真={rgb['distortion']}")
    if calib_date:
        print(f"  标定日期: {calib_date}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 rosbag2 提取相机内参生成 camera_calib.yaml")
    parser.add_argument("bag_dir", nargs="?", default="raw/session_001")
    parser.add_argument("output_dir", nargs="?", default="datasets/scenes1/session_001")
    args = parser.parse_args()
    sys.exit(extract_calib(args.bag_dir, args.output_dir))
