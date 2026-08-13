#!/usr/bin/env python3
"""
Convert memory service graph_store.json → ScribeMem-Bench memory_nodes.json.

The memory service (memgraph) produces graph_store.json during patrol via
the Scene Hook → remember pipeline.  This script converts those nodes to
the target dataset schema defined in 巡检数据集修改意见.md 附录 C.

Mapping of key fields:

  graph_store.json              →  memory_nodes.json (target)
  ─────────────────────────────────────────────────────────────
  summary (en, single)          →  summary_en  (kept as-is)
                                →  summary_zh  (placeholder, needs VLM)
  timestamp                     →  timestamp
  (derived)                     →  time_range {start_ts, end_ts}
  spatial_data.objects[].label  →  label_en  (kept as-is)
                                →  label_zh  (placeholder, needs VLM)
  tags.intent                   →  tags.intent_en
                                →  tags.intent_zh
  node_type ("short_term")      →  node_type ("ShortTerm")
  image_refs (memory svc paths) →  image_refs (rosbag frame paths, matched by ts)
  —                             →  depth_refs (matched by ts)
  —                             →  frame_ts[]
  —                             →  camera_params (from calib + odom pose)
  —                             →  video_clip_refs[]
  —                             →  confidence_flags{}
  raw_log, embedding, created_at, last_access →  (removed)

Usage:
  python3 scripts/convert_memory_nodes.py \
    --graph raw/session_001/memory/graph_store.json \
    --calib datasets/scenes1/session_001/camera_calib.yaml \
    --images-csv datasets/scenes1/session_001/images_index.csv \
    --depth-csv datasets/scenes1/session_001/depth_index.csv \
    --odom-csv datasets/scenes1/session_001/odometry.csv \
    --session-id session_001 \
    --output datasets/scenes1/session_001/memory_nodes.json
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None


# ── Node type mapping ───────────────────────────────────────────────────

_NODE_TYPE_MAP = {
    "short_term": "ShortTerm",
    "long_term": "LongTerm",
    "skill": "Skill",
    "fixed": "Fixed",
    "lesson": "Lesson",
}


def _map_node_type(raw: str) -> str:
    return _NODE_TYPE_MAP.get(raw, "ShortTerm")


# ── CSV helpers ──────────────────────────────────────────────────────────

def _read_csv_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def _find_nearest_frame(
    target_ts_ns: int,
    rows: list[dict],
    used: set,
    ts_col: str = "ts",
    filename_col: str = "file_path",
) -> Optional[dict]:
    """Find the unused frame closest in time to target_ts_ns."""
    best = None
    best_diff = float("inf")
    for row in rows:
        key = row.get(filename_col, "")
        if key in used:
            continue
        diff = abs(int(float(row[ts_col])) - target_ts_ns)
        if diff < best_diff:
            best_diff = diff
            best = row
    if best:
        used.add(best.get(filename_col, ""))
    return best


# ── Pose lookup ──────────────────────────────────────────────────────────

def _find_nearest_pose(
    target_ts_ns: int,
    poses: list[dict],
) -> Optional[dict]:
    """Find the base_link_pose closest in time (uses 't' column)."""
    best = None
    best_diff = float("inf")
    for p in poses:
        diff = abs(int(float(p.get("t", p.get("timestamp_ns", 0)))) - target_ts_ns)
        if diff < best_diff:
            best_diff = diff
            best = p
    return best


def _euler_to_quat(roll: float, pitch: float, yaw: float) -> dict:
    """Convert Euler angles (rad) to quaternion {qx, qy, qz, qw}."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    return {
        "qx": sr * cp * cy - cr * sp * sy,
        "qy": cr * sp * cy + sr * cp * sy,
        "qz": cr * cp * sy - sr * sp * cy,
        "qw": cr * cp * cy + sr * sp * sy,
    }


def _quat_to_yaw(rx: float, ry: float, rz: float, rw: float) -> float:
    """Extract yaw from quaternion."""
    siny_cosp = 2.0 * (rw * rz + rx * ry)
    cosy_cosp = 1.0 - 2.0 * (rz * rz + rx * rx)
    return math.atan2(siny_cosp, cosy_cosp)


# ── YOLO class → {category, label_zh} mapping ────────────────────────────

_CATEGORY_MAP = {
    "fire_extinguisher":   ("safety_equipment", "灭火器"),
    "exit_sign":           ("safety_equipment", "安全出口标识"),
    "potted_plant":        ("plant", "盆栽绿植"),
    "plant":               ("plant", "绿植"),
    "chair":               ("furniture", "椅子"),
    "table":               ("furniture", "桌子"),
    "desk":                ("furniture", "办公桌"),
    "cabinet":             ("furniture", "柜子"),
    "shelf":               ("furniture", "架子"),
    "couch":               ("furniture", "沙发"),
    "monitor":             ("it_equipment", "显示器"),
    "keyboard":            ("it_equipment", "键盘"),
    "laptop":              ("it_equipment", "笔记本电脑"),
    "mouse":               ("it_equipment", "鼠标"),
    "server_rack":         ("it_equipment", "服务器机柜"),
    "water_dispenser":     ("appliance", "饮水机"),
    "coffee_machine":      ("appliance", "咖啡机"),
    "refrigerator":        ("appliance", "冰箱"),
    "microwave":           ("appliance", "微波炉"),
    "water_bottle":        ("container", "水瓶"),
    "mug":                 ("container", "杯子"),
    "cup":                 ("container", "杯子"),
    "box":                 ("container", "盒子"),
    "cardboard_box":       ("container", "纸箱"),
    "thermos":             ("container", "保温杯"),
    "handbag":             ("personal_item", "手提包"),
    "backpack":            ("personal_item", "背包"),
    "lamp":                ("fixture", "台灯"),
    "light":               ("fixture", "灯"),
    "clock":               ("fixture", "钟"),
    "picture_frame":       ("decoration", "相框"),
    "vase":                ("decoration", "花瓶"),
    "whiteboard":          ("office_supply", "白板"),
    "trash_can":           ("fixture", "垃圾桶"),
    "door":                ("fixture", "门"),
    "window":              ("fixture", "窗户"),
    "phone":               ("personal_item", "手机"),
    "cell_phone":          ("personal_item", "手机"),
    "book":                ("office_supply", "书本"),
    "bottle":              ("container", "瓶子"),
    "person":              ("human", "人"),
    "car":                 ("vehicle", "汽车"),
    "bicycle":             ("vehicle", "自行车"),
    "tv":                  ("it_equipment", "电视"),
    "remote":              ("it_equipment", "遥控器"),
    "camera":              ("it_equipment", "相机"),
    "backpack":            ("personal_item", "背包"),
    "bag":                 ("personal_item", "包"),
    "umbrella":            ("personal_item", "雨伞"),
    "sink":                ("fixture", "水槽"),
    "toilet":              ("fixture", "马桶"),
    "bed":                 ("furniture", "床"),
    "bench":               ("furniture", "长凳"),
    "suitcase":            ("personal_item", "行李箱"),
    "shoe":                ("personal_item", "鞋子"),
}


def _lookup_category(cls_name: str) -> tuple:
    """Map YOLO class name → (category, label_zh)."""
    return _CATEGORY_MAP.get(cls_name.lower(), ("", cls_name))


# ── Main conversion ──────────────────────────────────────────────────────

def convert(
    graph_path: str,
    calib_path: str,
    images_csv: str,
    depth_csv: str,
    odom_csv: str,
    session_id: str,
    output_path: str,
) -> int:
    """Run the full conversion and write memory_nodes.json."""

    # ── Load inputs ──────────────────────────────────────────────────
    with open(graph_path, "r") as f:
        graph = json.load(f)

    calib = {}
    if yaml and os.path.exists(calib_path):
        with open(calib_path, "r") as f:
            calib = yaml.safe_load(f) or {}

    rgb_calib = calib.get("rgb", {})
    depth_calib = calib.get("depth", {})
    extrinsics = calib.get("extrinsics", {})

    images_idx = _read_csv_rows(images_csv)
    depth_idx = _read_csv_rows(depth_csv)
    odom_rows = _read_csv_rows(odom_csv)  # velocity-only per spec

    has_depth = len(depth_idx) > 0
    has_odom = len(odom_rows) > 0

    # 从 base_link_pose 读取轨迹位姿（用于 camera_pose）
    pose_csv = os.path.join(os.path.dirname(odom_csv), "base_link_pose.csv")
    pose_rows = _read_csv_rows(pose_csv)
    has_pose = len(pose_rows) > 0

    # 从 videos_index 读取视频流信息（用于 video_clip_refs）
    videos_csv = os.path.join(os.path.dirname(odom_csv), "videos_index.csv")
    video_refs_available = []
    if os.path.exists(videos_csv):
        video_refs_available = _read_csv_rows(videos_csv)

    print(f"graph_store: {len(graph.get('nodes', []))} nodes")
    print(f"images_idx:  {len(images_idx)} frames")
    print(f"depth_idx:   {len(depth_idx)} frames")
    print(f"odometry:    {len(odom_rows)} rows")
    print(f"base_link:   {len(pose_rows)} poses")
    print(f"calib:       {'loaded' if calib else 'MISSING'}")
    print(f"session_id:  {session_id}")

    # ── Convert each node ────────────────────────────────────────────
    used_images: set = set()
    used_depths: set = set()

    out_nodes = []
    for raw_node in graph.get("nodes", []):
        ts = raw_node.get("timestamp", 0)
        node_id = raw_node.get("node_id", 0)

        # ── Match image/depth frames by timestamp ─────────────────
        img_row = _find_nearest_frame(ts, images_idx, used_images)
        depth_row = None
        if has_depth:
            depth_row = _find_nearest_frame(ts, depth_idx, used_depths)

        # ── Build image_refs / depth_refs / frame_ts ───────────────
        image_refs = []
        depth_refs = []
        frame_ts = []

        if img_row:
            image_refs.append(f"{session_id}/{img_row['file_path']}")
            frame_ts.append(int(float(img_row["ts"])))

        if depth_row:
            depth_refs.append(f"{session_id}/{depth_row['file_path']}")
            # Align frame_ts with the image timestamp
            if not frame_ts:
                frame_ts.append(int(float(depth_row["ts"])))

        # ── Camera params from calib + odom pose ────────────────────
        camera_params = None
        if calib:
            # Default camera pose from extrinsics
            cp = {
                "x": extrinsics.get("rgb_to_base", {}).get("x", 0.15),
                "y": extrinsics.get("rgb_to_base", {}).get("y", 0.0),
                "z": extrinsics.get("rgb_to_base", {}).get("z", 0.85),
                "qx": extrinsics.get("rgb_to_base", {}).get("qx", 0.0),
                "qy": extrinsics.get("rgb_to_base", {}).get("qy", 0.0),
                "qz": extrinsics.get("rgb_to_base", {}).get("qz", 0.0),
                "qw": extrinsics.get("rgb_to_base", {}).get("qw", 1.0),
            }

            # If we have base_link trajectory, get actual robot pose at this timestamp
            if has_pose:
                pose = _find_nearest_pose(ts, pose_rows)
                if pose:
                    quat = _euler_to_quat(
                        float(pose.get("roll", 0.0)),
                        float(pose.get("pitch", 0.0)),
                        float(pose.get("yaw", 0.0)),
                    )
                    cp = {
                        "x": float(pose.get("x", 0.0)),
                        "y": float(pose.get("y", 0.0)),
                        "z": float(pose.get("z", 0.0)),
                        **quat,
                    }

            camera_params = {
                "fx": float(rgb_calib.get("fx", 525.0)),
                "fy": float(rgb_calib.get("fy", 525.0)),
                "cx": float(rgb_calib.get("cx", 319.5)),
                "cy": float(rgb_calib.get("cy", 239.5)),
                "width": int(rgb_calib.get("width", 640)),
                "height": int(rgb_calib.get("height", 480)),
                "distortion": [float(x) for x in rgb_calib.get("distortion", [0.0] * 5)],
                "camera_pose": cp,
                "depth_scale": float(depth_calib.get("depth_scale", 0.001)),
                "camera_type": "rgb",
            }

        # ── Derive intent (before tags mapping) ──────────────────
        raw_tags = raw_node.get("tags") or {}
        intent_from_svc = raw_tags.get("intent", "")
        if not intent_from_svc:
            # Try to derive from spatial objects
            raw_objs_preview = (raw_node.get("spatial_data") or {}).get("objects", [])
            if raw_objs_preview:
                obj_cls_names = [
                    _lookup_category(o.get("label", o.get("cls", "")))[1]
                    for o in raw_objs_preview[:3]
                ]
                intent_from_svc = f"观察 {'，'.join(obj_cls_names) if obj_cls_names else '周围环境'}"

        # ── Map tags ────────────────────────────────────────────────
        tags = {
            "scene_type": raw_tags.get("scene_type", ""),
            "objects_present": raw_tags.get("objects_present", []),
            "action_type": raw_tags.get("action_type", "observe"),
            "success": raw_tags.get("success", True),
            "task_type": raw_tags.get("task_type", "inspection"),
            "difficulty": raw_tags.get("difficulty", "easy"),
            "intent_zh": raw_tags.get("intent", "") or intent_from_svc,
            "intent_en": raw_tags.get("intent", "") or intent_from_svc,
        }

        # ── Map spatial_data.objects ────────────────────────────────
        raw_spatial = raw_node.get("spatial_data") or {}
        raw_objects = raw_spatial.get("objects", [])
        mapped_objects = []
        for obj in raw_objects:
            cls_name = obj.get("label", obj.get("cls", ""))
            category, label_zh = _lookup_category(cls_name)
            mapped_objects.append({
                "obj_id": obj.get("obj_id", ""),
                "label_zh": label_zh,
                "label_en": obj.get("label", cls_name),
                "x": float(obj.get("x", 0.0)),
                "y": float(obj.get("y", 0.0)),
                "z": float(obj.get("z", 0.0)),
            })

        spatial_data = {
            "objects": mapped_objects,
            "origin": raw_spatial.get("origin", "world"),
        }

        # ── Build target node ───────────────────────────────────────
        summary = raw_node.get("summary", "")
        causal_chain = raw_node.get("causal_chain", [])

        # video_clip_refs: 引用对应的视频流
        video_refs = [
            f"{session_id}/videos/{v.get('file_path', '').split('/')[-1]}"
            for v in video_refs_available
        ]

        node = {
            "node_id": node_id,
            "summary_zh": summary,              # placeholder: same as en for now
            "summary_en": summary,
            "timestamp": ts,
            "time_range": {
                "start_ts": ts,
                "end_ts": ts + int(0.5 * 1e9),  # assume ~0.5s observation window
            },
            "node_type": _map_node_type(raw_node.get("node_type", "short_term")),
            "spatial_data": spatial_data,
            "camera_params": camera_params,
            "tags": tags,
            "causal_chain": causal_chain,
            "weight": float(raw_node.get("weight", 0.5)),
            "image_refs": image_refs,
            "depth_refs": depth_refs,
            "frame_ts": frame_ts,
            "video_clip_refs": video_refs,
            "confidence_flags": {
                "object_detection": 0.5,
                "tag_classification": 0.5,
            },
            "access_count": raw_node.get("access_count", 0),
            "version": raw_node.get("version", 1),
        }
        out_nodes.append(node)

    # ── Write output ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_nodes, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 转换完成: {len(out_nodes)} nodes → {output_path}")
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert graph_store.json → ScribeMem-Bench memory_nodes.json"
    )
    parser.add_argument("--graph", required=True, help="path to graph_store.json")
    parser.add_argument("--calib", required=True, help="path to camera_calib.yaml")
    parser.add_argument("--images-csv", required=True, help="path to images_index.csv")
    parser.add_argument("--depth-csv", default="", help="path to depth_index.csv")
    parser.add_argument("--odom-csv", default="", help="path to odometry.csv (for camera pose)")
    parser.add_argument("--session-id", required=True, help="e.g. session_001")
    parser.add_argument("--output", required=True, help="output memory_nodes.json path")
    args = parser.parse_args()

    return convert(
        graph_path=args.graph,
        calib_path=args.calib,
        images_csv=args.images_csv,
        depth_csv=args.depth_csv,
        odom_csv=args.odom_csv,
        session_id=args.session_id,
        output_path=args.output,
    )


if __name__ == "__main__":
    sys.exit(main())
