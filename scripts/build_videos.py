#!/usr/bin/env python3
"""
从已提取的图像/深度帧序列编码视频流（附录 H 格式）。

输入:
  output_dir/images/          RGB 帧 (PNG)
  output_dir/depth/           深度帧 (PNG, uint16 mm)
  output_dir/images_index.csv
  output_dir/depth_index.csv

输出:
  output_dir/videos/rgb_main.mp4       H.264 编码 RGB 视频
  output_dir/videos/depth_aligned.mp4  H.264 编码深度视频 (uint16 双通道 BGR 打包)
  output_dir/videos_index.csv          视频流索引

用法:
  python3 scripts/build_videos.py datasets/scenes1/session_001
  python3 scripts/build_videos.py datasets/scenes1/session_001 --no-depth
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def _read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def _get_resolution(rows: list[dict], images_dir: str) -> tuple[int, int]:
    """从第一帧图片读取实际分辨率。"""
    if not rows:
        return 640, 480
    first_path = os.path.join(os.path.dirname(images_dir), rows[0]["file_path"])
    if os.path.exists(first_path):
        img = cv2.imread(first_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img.shape[1], img.shape[0]
    return 640, 480


def build_rgb_video(
    images_dir: str,
    images_index: list[dict],
    output_video: str,
    fps: float = 15.0,
) -> dict | None:
    """
    将 RGB 帧序列编码为 H.264 MP4。

    Returns:
        videos_index 条目，失败返回 None
    """
    if not images_index:
        print("  ⚠ 无 RGB 帧，跳过视频编码")
        return None

    width, height = _get_resolution(images_index, images_dir)

    # OpenH264 编码器 (mp4v + avc1)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    if fourcc == 0:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    os.makedirs(os.path.dirname(output_video), exist_ok=True)

    writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    if not writer.isOpened():
        # 回退到 mp4v
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    if not writer.isOpened():
        print("  ✗ 无法创建视频编码器（缺少 codec）", file=sys.stderr)
        return None

    session_dir = os.path.dirname(images_dir)  # output_dir

    count = 0
    start_ts = None
    end_ts = None
    start_frame = None
    end_frame = None

    for row in images_index:
        file_path = os.path.join(session_dir, row["file_path"])
        if not os.path.exists(file_path):
            continue

        img = cv2.imread(file_path)
        if img is None:
            continue

        ts = int(float(row["ts"]))
        frame_id = int(row["frame_id"])

        if start_ts is None:
            start_ts = ts
            start_frame = frame_id
        end_ts = ts
        end_frame = frame_id

        writer.write(img)
        count += 1

        if count % 200 == 0:
            print(f"  RGB 视频: 已编码 {count} 帧...")

    writer.release()

    if count == 0:
        print("  ✗ 未能编码任何 RGB 帧", file=sys.stderr)
        return None

    duration = (end_ts - start_ts) / 1e9 if start_ts and end_ts else 0

    print(f"✓ RGB 视频: {count} 帧, {duration:.1f}s → {output_video}")

    return {
        "video_id": "",
        "file_path": os.path.relpath(output_video, session_dir),
        "camera_type": "rgb",
        "codec": "h264",
        "resolution": f"{width}×{height}",
        "fps": fps,
        "duration_sec": round(duration, 1),
        "start_ts": start_ts or 0,
        "end_ts": end_ts or 0,
        "start_frame_id": start_frame or 0,
        "end_frame_id": end_frame or 0,
        "count": count,
    }


def pack_depth_u16_to_bgr(img_u16: "np.ndarray") -> "np.ndarray":
    """
    将 uint16 深度图打包为 3 通道 BGR 图像（无损）。

    OpenCV VideoWriter 不支持 uint16 单通道，因此将 uint16 拆分为:
      B 通道 = 低 8 位 (bits 0-7)
      G 通道 = 高 8 位 (bits 8-15)
      R 通道 = 0 (保留)

    解码还原: depth = (G << 8) | B
    """
    low = (img_u16 & 0xFF).astype(np.uint8)
    high = ((img_u16 >> 8) & 0xFF).astype(np.uint8)
    packed = np.zeros((img_u16.shape[0], img_u16.shape[1], 3), dtype=np.uint8)
    packed[:, :, 0] = low   # B
    packed[:, :, 1] = high  # G
    return packed


def unpack_depth_bgr_to_u16(frame_bgr: "np.ndarray") -> "np.ndarray":
    """从 BGR 帧还原 uint16 深度图（pack_depth_u16_to_bgr 的逆操作）。"""
    low = frame_bgr[:, :, 0].astype(np.uint16)
    high = frame_bgr[:, :, 1].astype(np.uint16)
    return (high << 8) | low


def build_depth_video(
    depth_dir: str,
    depth_index: list[dict],
    output_video: str,
    fps: float = 15.0,
) -> dict | None:
    """
    将深度帧序列编码为无损视频（uint16 → 双通道 BGR 打包）。

    深度值单位: mm (uint16)，拆分为 B=低8位, G=高8位, R=0，
    解码时用 unpack_depth_bgr_to_u16() 还原完整 uint16 精度。

    编码器优先级: FFV1 (MKV) > MPNG (AVI) > HFYU (AVI)
    全部为无损编码，roundtrip 后像素值完全一致。
    """
    if not depth_index:
        print("  ⚠ 无深度帧，跳过深度视频编码")
        return None

    width, height = _get_resolution(depth_index, depth_dir)

    # 无损编码器优先级（depth 数据必须无损）
    LOSSLESS_CODECS = [
        ("FFV1", ".mkv"),
        ("MPNG", ".avi"),
        ("HFYU", ".avi"),
    ]

    writer = None
    actual_ext = ".mp4"
    used_codec = ""

    for codec_str, ext in LOSSLESS_CODECS:
        fourcc = cv2.VideoWriter_fourcc(*codec_str)
        test_path = output_video.replace(".mp4", ext).replace(".avi", ext)
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        w = cv2.VideoWriter(test_path, fourcc, fps, (width, height), isColor=True)
        if w.isOpened():
            writer = w
            actual_ext = ext
            used_codec = codec_str
            output_video = test_path
            break
        w.release()

    if writer is None:
        print("  ✗ 无法创建深度视频编码器（缺少无损 codec）", file=sys.stderr)
        return None

    print(f"  深度编码器: {used_codec} ({actual_ext})")

    session_dir = os.path.dirname(depth_dir)  # output_dir

    count = 0
    start_ts = None
    end_ts = None
    start_frame = None
    end_frame = None

    for row in depth_index:
        file_path = os.path.join(session_dir, row["file_path"])
        if not os.path.exists(file_path):
            continue

        # 读 16-bit 深度 PNG
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        # 确保是 2D 数组
        if img.ndim == 3:
            img = img[:, :, 0]

        # 打包 uint16 → BGR 三通道
        packed = pack_depth_u16_to_bgr(img)

        ts = int(float(row["ts"]))
        frame_id = int(row["frame_id"])

        if start_ts is None:
            start_ts = ts
            start_frame = frame_id
        end_ts = ts
        end_frame = frame_id

        writer.write(packed)
        count += 1

        if count % 200 == 0:
            print(f"  深度视频: 已编码 {count} 帧...")

    writer.release()

    if count == 0:
        print("  ✗ 未能编码任何深度帧", file=sys.stderr)
        return None

    duration = (end_ts - start_ts) / 1e9 if start_ts and end_ts else 0

    print(f"✓ 深度视频: {count} 帧 (无损 {used_codec}), {duration:.1f}s → {output_video}")

    return {
        "video_id": "",
        "file_path": os.path.relpath(output_video, session_dir),
        "camera_type": "depth",
        "codec": f"{used_codec}_uint16_packed",
        "resolution": f"{width}×{height}",
        "fps": fps,
        "duration_sec": round(duration, 1),
        "start_ts": start_ts or 0,
        "end_ts": end_ts or 0,
        "start_frame_id": start_frame or 0,
        "end_frame_id": end_frame or 0,
        "count": count,
    }


def write_videos_index(output_dir: str, entries: list[dict], session_id: str) -> None:
    """写入 videos_index.csv（附录 H 格式）。"""
    index_path = os.path.join(output_dir, "videos_index.csv")

    # 给每个 entry 赋 video_id，file_path 已在上游填充为相对路径
    camera_suffix = {"rgb": "main", "depth": "aligned", "thermal": "thermal"}
    for e in entries:
        if not e:
            continue
        suffix = camera_suffix.get(e["camera_type"], e["camera_type"])
        e["video_id"] = f"vid_{session_id}_{e['camera_type']}_{suffix}"

    with open(index_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "video_id", "file_path", "camera_type", "codec",
            "resolution", "fps", "duration_sec",
            "start_ts", "end_ts", "start_frame_id", "end_frame_id",
        ])
        for e in entries:
            if not e:
                continue
            writer.writerow([
                e["video_id"], e["file_path"], e["camera_type"], e["codec"],
                e["resolution"], e["fps"], e["duration_sec"],
                e["start_ts"], e["end_ts"], e["start_frame_id"], e["end_frame_id"],
            ])

    print(f"✓ 视频索引: {len([e for e in entries if e])} 条 → {index_path}")


def update_meta_video_config(output_dir: str) -> None:
    """更新 meta.yaml 中的 video 配置为 enabled。"""
    try:
        import yaml
    except ImportError:
        return

    meta_path = os.path.join(output_dir, "meta.yaml")
    if not os.path.exists(meta_path):
        return

    with open(meta_path, "r") as f:
        meta = yaml.safe_load(f) or {}

    sensors = meta.get("sensor_configs", {})
    video_cfg = sensors.get("video", {})
    video_cfg["enabled"] = True
    if "codec" not in video_cfg:
        video_cfg["codec"] = "h264"
    if "resolution" not in video_cfg:
        video_cfg["resolution"] = "640×480"
    if "fps" not in video_cfg:
        video_cfg["fps"] = 15.0
    if "sample_fps" not in video_cfg:
        video_cfg["sample_fps"] = 1.0
    sensors["video"] = video_cfg
    meta["sensor_configs"] = sensors

    with open(meta_path, "w") as f:
        yaml.safe_dump(meta, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print(f"✓ meta.yaml video 配置已更新")


def main():
    parser = argparse.ArgumentParser(description="从图像帧序列编码视频流")
    parser.add_argument("output_dir", help="session 输出目录")
    parser.add_argument("--fps", type=float, default=15.0, help="视频帧率 (默认 15)")
    parser.add_argument("--no-depth", action="store_true", help="跳过深度视频编码")
    parser.add_argument("--no-rgb", action="store_true", help="跳过 RGB 视频编码")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    session_id = os.path.basename(output_dir)  # e.g. "session_001"
    videos_dir = os.path.join(output_dir, "videos")

    images_csv = os.path.join(output_dir, "images_index.csv")
    depth_csv = os.path.join(output_dir, "depth_index.csv")

    images_index = sorted(
        _read_csv(images_csv),
        key=lambda r: int(float(r["ts"])),
    )
    depth_index = sorted(
        _read_csv(depth_csv),
        key=lambda r: int(float(r["ts"])),
    )

    print(f"RGB 帧: {len(images_index)}, 深度帧: {len(depth_index)}")

    entries = []

    # RGB 视频
    if not args.no_rgb and images_index:
        rgb_video_path = os.path.join(videos_dir, "rgb_main.mp4")
        entry = build_rgb_video(
            os.path.join(output_dir, "images"),
            images_index,
            rgb_video_path,
            fps=args.fps,
        )
        entries.append(entry)

    # 深度视频
    if not args.no_depth and depth_index:
        depth_video_path = os.path.join(videos_dir, "depth_aligned.mp4")
        entry = build_depth_video(
            os.path.join(output_dir, "depth"),
            depth_index,
            depth_video_path,
            fps=args.fps,
        )
        entries.append(entry)

    # 写索引 & 更新 meta
    valid = [e for e in entries if e]
    if valid:
        write_videos_index(output_dir, valid, session_id)
        update_meta_video_config(output_dir)
    else:
        print("  ⓘ 未生成任何视频")


if __name__ == "__main__":
    main()
