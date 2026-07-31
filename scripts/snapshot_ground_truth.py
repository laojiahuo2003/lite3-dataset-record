#!/usr/bin/env python3
"""
从运行中的 Robonix 服务抓取 Ground Truth 快照（无需人工填写）。

数据来源：
  - scene 服务   (默认 http://127.0.0.1:50107/api/state)
      YOLO-World 开集检测的物体列表 + 房间标注多边形 + 机器人位姿
  - mapping 服务 (默认 http://127.0.0.1:8091)
      已保存地图的 occupancy 平面图

输出（最终数据集都在 datasets/ 下；地图为锚，多个 session 共享）：
  datasets/session_XXX/objects.yaml    物体标注（含 map_id）
  datasets/scenes/<map_id>/scene.yaml  场景/房间定义
  datasets/scenes/<map_id>/layout.png  平面图（若地图已保存）
  raw/session_XXX/meta.yaml            写入 map_id 字段（session↔map 绑定源头）

用法：
  python3 scripts/snapshot_ground_truth.py 003
  python3 scripts/snapshot_ground_truth.py 003 --min-conf 0.3
"""

import argparse
import json
import shutil
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SCENE_URL = "http://127.0.0.1:50107/api/state"
DEFAULT_MAPPING_URL = "http://127.0.0.1:8091"
DEFAULT_MAPS_DIR = Path.home() / ".robonix" / "maps"


def _get_json(url: str, timeout: float = 8.0):
    """GET 一个 JSON 接口，失败返回 None（服务没起来时不阻塞）。"""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ 无法访问 {url}: {exc}", file=sys.stderr)
        return None


def _point_in_polygon(x: float, y: float, poly: list) -> bool:
    """射线法判断点 (x, y) 是否落在多边形内。"""
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i][0], poly[i][1]
        xj, yj = poly[j][0], poly[j][1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def _room_of(x: float, y: float, rooms: list) -> str:
    """返回物体所在房间名，找不到返回空串。"""
    for room in rooms:
        if room["kind"] == "room" and _point_in_polygon(x, y, room["points"]):
            return room["name"]
    return ""


def build_objects_gt(state: dict, session_id: str, min_conf: float) -> dict:
    """从 scene 状态抽出物体标注。"""
    rooms = [a for a in state.get("annotations", []) if a.get("kind") == "room"]
    objects = []
    skipped = 0
    for obj in state.get("objects", []):
        if obj.get("cls") == "robot":
            continue
        conf = float(obj.get("confidence", 0.0))
        if conf < min_conf:
            skipped += 1
            continue
        pose = obj.get("pose", {})
        bbox = obj.get("bbox", {})
        objects.append({
            "id": obj.get("short_id") or obj.get("id"),
            "class": obj.get("cls"),
            "location": _room_of(pose.get("x", 0.0), pose.get("y", 0.0), rooms),
            "pose": {k: round(float(pose.get(k, 0.0)), 4) for k in ("x", "y", "z", "yaw")},
            "size": {
                "x": round(float(bbox.get("size_x", 0.0)), 4),
                "y": round(float(bbox.get("size_y", 0.0)), 4),
                "z": round(float(bbox.get("size_z", 0.0)), 4),
            },
            "confidence": round(conf, 4),
            "observation_count": int(obj.get("observation_count", 0)),
            "missing": bool(obj.get("missing", False)),
        })
    objects.sort(key=lambda o: (o["location"], o["class"], o["id"]))
    return {
        "session_id": session_id,
        "captured_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "robonix scene service (YOLO-World open-set detection)",
        "map_id": (state.get("map_binding") or {}).get("map_id", ""),
        "min_confidence": min_conf,
        "object_count": len(objects),
        "skipped_low_confidence": skipped,
        "objects": objects,
    }


def build_scene_def(state: dict, maps_dir: Path) -> dict:
    """从 scene 状态 + 地图元信息构建场景定义。"""
    binding = state.get("map_binding") or {}
    map_id = binding.get("map_id", "")
    rooms = [
        {"name": a.get("name"), "kind": a.get("kind"), "points": a.get("points")}
        for a in state.get("annotations", [])
        if a.get("kind") == "room"
    ]
    scene = {
        "map_id": map_id,
        "captured_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "robot_pose": state.get("robot"),
        "rooms": rooms,
    }
    meta_path = maps_dir / map_id / "meta.yaml"
    if meta_path.exists():
        try:
            scene["map_meta"] = yaml.safe_load(meta_path.read_text())
        except Exception:  # noqa: BLE001
            pass
    return scene


def copy_layout(map_id: str, maps_dir: Path, scene_dir: Path) -> bool:
    """把已保存地图的 occupancy.png 拷成 scenes/<map_id>/layout.png。"""
    src = maps_dir / map_id / "occupancy.png"
    if not map_id or not src.exists():
        return False
    scene_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, scene_dir / "layout.png")
    return True


def _dump_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def bind_session_to_map(project: Path, session_id: str, map_id: str) -> str:
    """把 map_id 写进会话的 raw/session_XXX/meta.yaml，建立显式绑定。

    返回一条状态说明。若 meta.yaml 已有不同的 map_id，保留原值并告警
    （同一 session 不应跨地图）。
    """
    meta_path = project / "raw" / f"session_{session_id}" / "meta.yaml"
    if not meta_path.exists():
        return f"  ⓘ 未找到 {meta_path}，跳过绑定（录制尚未生成 meta.yaml？）"
    try:
        meta = yaml.safe_load(meta_path.read_text()) or {}
    except Exception:  # noqa: BLE001
        return f"  ⚠ 无法解析 {meta_path}，跳过 map_id 绑定"
    existing = meta.get("map_id")
    if existing and existing != map_id:
        return (f"  ⚠ meta.yaml 已绑定 map_id='{existing}'，与当前 scene 的 "
                f"'{map_id}' 不一致；保留原值。请确认定位模式用的是哪张地图")
    meta["map_id"] = map_id
    _dump_yaml(meta_path, meta)
    return f"✓ 会话已绑定地图 map_id='{map_id}' → {meta_path}"


def main() -> int:
    ap = argparse.ArgumentParser(description="抓取 Robonix 服务的 Ground Truth 快照")
    ap.add_argument("session", help="session 编号，如 003")
    ap.add_argument("--scene-url", default=DEFAULT_SCENE_URL)
    ap.add_argument("--mapping-url", default=DEFAULT_MAPPING_URL)
    ap.add_argument("--maps-dir", default=str(DEFAULT_MAPS_DIR))
    ap.add_argument("--min-conf", type=float, default=0.0, help="物体置信度下限，默认 0")
    ap.add_argument("--project-dir", default=str(PROJECT_DIR))
    args = ap.parse_args()

    session_id = args.session.zfill(3) if args.session.isdigit() else args.session
    project = Path(args.project_dir)
    maps_dir = Path(args.maps_dir)

    print("抓取 scene 状态 ...")
    state = _get_json(args.scene_url)
    if state is None:
        print("✗ scene 服务不可达；请确认机器狗已 ./start.sh 且 scene 已就绪", file=sys.stderr)
        return 1

    objects_gt = build_objects_gt(state, session_id, args.min_conf)
    obj_path = project / "datasets" / f"session_{session_id}" / "objects.yaml"
    _dump_yaml(obj_path, objects_gt)
    print(f"✓ 物体标注 {objects_gt['object_count']} 个 → {obj_path}")

    # 一张地图一个目录，多个 session 共享
    scene_def = build_scene_def(state, maps_dir)
    map_id = scene_def["map_id"] or "unknown_map"
    scene_dir = project / "datasets" / "scenes" / map_id
    _dump_yaml(scene_dir / "scene.yaml", scene_def)
    print(f"✓ 场景定义（{len(scene_def['rooms'])} 房间）→ {scene_dir / 'scene.yaml'}")

    if copy_layout(map_id, maps_dir, scene_dir):
        print(f"✓ 平面图 → {scene_dir / 'layout.png'}")
    else:
        print(f"  ⓘ 地图 '{map_id}' 尚未保存 occupancy.png，跳过平面图（房间多边形已在场景定义中）")

    # 把 session 显式绑定到这张地图
    print(bind_session_to_map(project, session_id, map_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
