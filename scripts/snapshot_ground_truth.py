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

# ── YOLO class → (category, label_zh) mapping ──────────────────────────
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
    "microwave":           ("appliance", "微波炉"),
    "phone":               ("personal_item", "手机"),
    "cell_phone":          ("personal_item", "手机"),
    "book":                ("office_supply", "书本"),
    "bottle":              ("container", "瓶子"),
    "person":              ("human", "人"),
    "robot":               ("robot", "机器人"),
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
    "pillow":              ("furniture", "枕头"),
    "bench":               ("furniture", "长凳"),
    "suitcase":            ("personal_item", "行李箱"),
    "shoe":                ("personal_item", "鞋子"),
}


def _lookup_category(cls_name: str) -> tuple:
    """Map YOLO class name → (category, label_zh)."""
    return _CATEGORY_MAP.get(cls_name.lower(), ("", cls_name))


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
    """从 scene 状态抽出物体标注，对齐改版 objects.yaml schema."""
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
        cls_name = obj.get("cls", "unknown")
        short_id = obj.get("short_id") or obj.get("id", "")
        category, label_zh = _lookup_category(cls_name)

        objects.append({
            "obj_id": f"scene.object.{cls_name}_{short_id.split('_')[-1] if '_' in short_id else short_id}",
            "label_zh": label_zh,                   # auto-mapped from category table
            "label_en": cls_name,
            "category": category,                    # auto-mapped from category table
            "position": {
                "x": round(float(pose.get("x", 0.0)), 4),
                "y": round(float(pose.get("y", 0.0)), 4),
                "z": round(float(pose.get("z", 0.0)), 4),
            },
            "bbox_3d": {
                "dx": round(float(bbox.get("size_x", 0.0)), 4),
                "dy": round(float(bbox.get("size_y", 0.0)), 4),
                "dz": round(float(bbox.get("size_z", 0.0)), 4),
            },
            "first_seen_ts": 0,
            "last_seen_ts": 0,
            "attributes": {
                "status": "missing" if bool(obj.get("missing", False)) else "normal",
                "state_detail": "",
            },
            "confidence": round(conf, 4),
            "observation_count": int(obj.get("observation_count", 0)),
            "location": _room_of(pose.get("x", 0.0), pose.get("y", 0.0), rooms),
        })
    objects.sort(key=lambda o: (o["location"], o["category"], o["label_en"]))
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


def _infer_region_type(name_zh: str) -> str:
    """Derive region type from Chinese room name."""
    type_hints = {
        "办公": "indoor_office", "会议": "meeting_room", "茶水": "kitchen",
        "走廊": "corridor", "走道": "corridor", "过道": "corridor",
        "大厅": "entrance", "入口": "entrance", "前台": "entrance",
        "实验": "lab", "机": "server_room", "仓库": "storage",
        "存储": "storage", "卫生": "restroom", "出口": "exit",
        "户外": "outdoor", "室外": "outdoor", "门": "entrance",
    }
    for keyword, rt in type_hints.items():
        if keyword in name_zh:
            return rt
    return ""


def build_scene_def(state: dict, maps_dir: Path) -> dict:
    """从 scene 状态 + 地图元信息构建改版场景定义."""
    binding = state.get("map_binding") or {}
    map_id = binding.get("map_id", "")
    raw_rooms = [
        a for a in state.get("annotations", [])
        if a.get("kind") == "room"
    ]

    # Build regions list with auto-computed bounds
    regions = []
    for i, room in enumerate(raw_rooms):
        pts = room.get("points", [])
        xs = [p[0] for p in pts] if pts else [0, 0]
        ys = [p[1] for p in pts] if pts else [0, 0]
        # Compute area via shoelace formula
        area = 0.0
        if len(pts) >= 3:
            for j in range(len(pts)):
                x1, y1 = pts[j][0], pts[j][1]
                x2, y2 = pts[(j + 1) % len(pts)][0], pts[(j + 1) % len(pts)][1]
                area += x1 * y2 - x2 * y1
            area = abs(area) / 2.0

        name_zh = room.get("name", f"区域{i + 1}")

        regions.append({
            "region_id": f"region_{i + 1}",
            "name_zh": name_zh,
            "name_en": f"Region {i + 1}",
            "type": _infer_region_type(name_zh),
            "area_m2": round(area, 2),
            "bounds": {
                "x_min": round(min(xs), 2),
                "x_max": round(max(xs), 2),
                "y_min": round(min(ys), 2),
                "y_max": round(max(ys), 2),
            },
            "adjacency": [],                          # needs manual fill
            "points": pts,
        })

    scene = {
        "scene_id": "scenes1",
        "scene_name_zh": map_id or "未命名场景",      # default from map_id
        "scene_name_en": map_id or "Unnamed Scene",
        "total_area_m2": round(sum(r["area_m2"] for r in regions), 2),
        "map_id": map_id,
        "captured_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "robot_pose": state.get("robot"),
        "regions": regions,
        "patrol_routes": [],                          # needs manual fill
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
    ap.add_argument("--scene", default="scenes1", help="场景目录名 (默认 scenes1)")
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
    scene_name = args.scene
    obj_path = project / "datasets" / scene_name / f"session_{session_id}" / "objects.yaml"
    _dump_yaml(obj_path, objects_gt)
    print(f"✓ 物体标注 {objects_gt['object_count']} 个 → {obj_path}")

    # 一张地图一个目录，多个 session 共享
    scene_def = build_scene_def(state, maps_dir)
    map_id = scene_def["map_id"] or "unknown_map"
    scene_dir = project / "datasets" / scene_name / map_id
    _dump_yaml(scene_dir / "scene.yaml", scene_def)
    print(f"✓ 场景定义（{len(scene_def['regions'])} 房间）→ {scene_dir / 'scene.yaml'}")

    if copy_layout(map_id, maps_dir, scene_dir):
        print(f"✓ 平面图 → {scene_dir / 'layout.png'}")
    else:
        print(f"  ⓘ 地图 '{map_id}' 尚未保存 occupancy.png，跳过平面图（房间多边形已在场景定义中）")

    # 把 session 显式绑定到这张地图
    print(bind_session_to_map(project, session_id, map_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
