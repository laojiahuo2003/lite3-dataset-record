#!/usr/bin/env python3
"""
ScribeMem-Bench 数据集构建主控脚本（改版格式）。

从 rosbag2 中提取全部传感器数据，并组装为结构化的记忆数据集。

用法:
  python3 scripts/assemble_dataset.py                    # 处理所有 session
  python3 scripts/assemble_dataset.py -s 001             # 指定单个
  python3 scripts/assemble_dataset.py -s 002 --skip-existing  # 增量
  python3 scripts/assemble_dataset.py -s 001 --only images,odom
  python3 scripts/assemble_dataset.py -s 001 --scene scenes2  # 指定场景

提取流程:
  Phase 1: 里程计提取  extract_odom.py   → odometry.csv
  Phase 2: TF 轨迹提取  extract_tf.py     → base_link_pose.csv + tf_trajectory.csv
  Phase 3: 图像提取    extract_images.py  → images/
  Phase 4: 深度图提取  extract_depth.py   → depth/
  Phase 5: 点云提取    extract_lidar.py   → lidar/
  Phase 6: 模板拷贝    camera_calib.yaml, events.yaml, videos_index.csv, meta.yaml
  Phase 7: 数据增强    enrich_images_csv, enrich_depth_csv
  Phase 8: 视频编码    build_videos.py → videos/
  Phase 9: 记忆节点    convert_memory_nodes.py 或 build_memory_nodes.py
  Phase 10: 清单生成   dataset_manifest.yaml（附录 G 格式）
"""

import argparse
import csv
import glob
import json as _json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_script(script_name: str, args: list[str], desc: str) -> int:
    """运行提取脚本并计时。"""
    script_path = os.path.join(SCRIPT_DIR, script_name)
    print(f"\n{'='*60}")
    print(f"  Phase: {desc}")
    print(f"  脚本: {script_name}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    result = subprocess.run([sys.executable, script_path] + args, check=False)
    elapsed = time.perf_counter() - t0

    if result.returncode == 0:
        print(f"  ✓ 完成 (耗时 {elapsed:.1f}s)")
    else:
        print(f"  ✗ 失败 (退出码 {result.returncode})", file=sys.stderr)
    return result.returncode


def check_exists(output_dir: str, files: list[str]) -> bool:
    """检查输出文件是否全部存在。"""
    return all(os.path.exists(os.path.join(output_dir, f)) for f in files)


# ── 数据增强 ──────────────────────────────────────────────────────────────

def enrich_images_csv(output_dir: str) -> None:
    """回填 images_index.csv 的相机位姿列（通过 ts 匹配 base_link_pose）。"""
    import math as _math

    images_csv = os.path.join(output_dir, "images_index.csv")
    pose_csv = os.path.join(output_dir, "base_link_pose.csv")

    if not os.path.exists(images_csv) or not os.path.exists(pose_csv):
        print("  ⏭ 跳过 images 位姿增强（缺少 csv）")
        return

    # 读取 base_link_pose
    pose_rows = []
    with open(pose_csv, "r") as f:
        for row in csv.DictReader(f):
            pose_rows.append(row)
    if not pose_rows:
        print("  ⏭ 跳过 images 位姿增强（base_link_pose 为空）")
        return

    # 读取 images_index
    with open(images_csv, "r") as f:
        reader = csv.DictReader(f)
        img_rows = list(reader)
        fieldnames = reader.fieldnames

    # 匹配每帧最近的 base_link pose (euler → quat)
    enriched = 0
    for img in img_rows:
        img_ts = int(float(img["ts"]))
        best = None
        best_diff = float("inf")
        for p in pose_rows:
            diff = abs(int(float(p["t"])) - img_ts)
            if diff < best_diff:
                best_diff = diff
                best = p
        if best:
            yaw = float(best.get("yaw", 0.0))
            img["cam_x"] = best.get("x", "0")
            img["cam_y"] = best.get("y", "0")
            img["cam_z"] = best.get("z", "0")
            img["cam_qx"] = "0.0"
            img["cam_qy"] = "0.0"
            img["cam_qz"] = str(_math.sin(yaw / 2.0))
            img["cam_qw"] = str(_math.cos(yaw / 2.0))
            enriched += 1

    with open(images_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(img_rows)

    print(f"  ✓ images 位姿增强: {enriched}/{len(img_rows)} 帧")


def enrich_depth_csv(output_dir: str) -> None:
    """回填 depth_index.csv 的 aligned_to_rgb_frame（通过 ts 最近匹配）。"""
    depth_csv = os.path.join(output_dir, "depth_index.csv")
    images_csv = os.path.join(output_dir, "images_index.csv")

    if not os.path.exists(depth_csv) or not os.path.exists(images_csv):
        print("  ⏭ 跳过 depth 对齐增强（缺少 csv）")
        return

    # 读取 images_index
    with open(images_csv, "r") as f:
        img_rows = list(csv.DictReader(f))

    if not img_rows:
        print("  ⏭ 跳过 depth 对齐增强（images 为空）")
        return

    # 读取 depth_index
    with open(depth_csv, "r") as f:
        reader = csv.DictReader(f)
        depth_rows = list(reader)
        fieldnames = reader.fieldnames

    # 匹配每帧深度最近的 RGB frame_id
    aligned = 0
    for depth in depth_rows:
        depth_ts = int(float(depth["ts"]))
        best = None
        best_diff = float("inf")
        for img in img_rows:
            diff = abs(int(float(img["ts"])) - depth_ts)
            if diff < best_diff:
                best_diff = diff
                best = img
        if best is not None:
            depth["aligned_to_rgb_frame"] = best["frame_id"]
            aligned += 1

    with open(depth_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(depth_rows)

    print(f"  ✓ depth 对齐增强: {aligned}/{len(depth_rows)} 帧")


# ── Manifest 生成 ─────────────────────────────────────────────────────────

def generate_manifest(output_dir: str, scene_id: str, session_id: str) -> None:
    """生成 dataset_manifest.yaml（附录 G 格式）。"""
    try:
        import yaml as _yaml
    except ImportError:
        print("  ⚠ 未安装 yaml，跳过 manifest 生成")
        return

    # 统计数据
    nodes_path = os.path.join(output_dir, "memory_nodes.json")
    num_nodes = 0
    if os.path.exists(nodes_path):
        with open(nodes_path, "r") as f:
            num_nodes = len(_json.load(f))

    images_csv = os.path.join(output_dir, "images_index.csv")
    num_images = sum(1 for _ in open(images_csv)) - 1 if os.path.exists(images_csv) else 0

    depth_csv = os.path.join(output_dir, "depth_index.csv")
    num_depth = sum(1 for _ in open(depth_csv)) - 1 if os.path.exists(depth_csv) else 0

    lidar_dir = os.path.join(output_dir, "lidar")
    num_lidar = len(glob.glob(os.path.join(lidar_dir, "*.pcd"))) if os.path.isdir(lidar_dir) else 0

    videos_csv = os.path.join(output_dir, "videos_index.csv")
    num_videos = sum(1 for _ in open(videos_csv)) - 1 if os.path.exists(videos_csv) else 0

    objects_yaml = os.path.join(output_dir, "objects.yaml")
    num_objects = 0
    if os.path.exists(objects_yaml):
        try:
            obj_data = _yaml.safe_load(open(objects_yaml))
            num_objects = obj_data.get("object_count", 0) if obj_data else 0
        except Exception:
            pass

    events_yaml = os.path.join(output_dir, "events.yaml")
    num_events = 0
    if os.path.exists(events_yaml):
        try:
            ev_data = _yaml.safe_load(open(events_yaml))
            num_events = len(ev_data.get("events", [])) if ev_data else 0
        except Exception:
            pass

    # 从元数据推断时段
    time_of_day = ""
    meta_yaml = os.path.join(output_dir, "meta.yaml")
    if os.path.exists(meta_yaml):
        try:
            meta = _yaml.safe_load(open(meta_yaml))
            lighting = (meta or {}).get("lighting_condition", "")
            if lighting:
                time_of_day = lighting  # daylight / artificial / night
        except Exception:
            pass

    # 从里程计计算实际时长
    duration_sec = 0.0
    odom_csv = os.path.join(output_dir, "odometry.csv")
    if os.path.exists(odom_csv):
        try:
            with open(odom_csv) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if len(rows) >= 2:
                t0 = int(float(rows[0].get("t", rows[0].get("timestamp_ns", 0))))
                t1 = int(float(rows[-1].get("t", rows[-1].get("timestamp_ns", 0))))
                duration_sec = round((t1 - t0) / 1e9, 1)
        except Exception:
            pass

    manifest = {
        "scene_id": scene_id,
        "session_id": session_id,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "time_of_day": time_of_day,
        "platform": "Deep Robotics Lite3 (quadruped robot dog)",
        "duration_sec": duration_sec,
        "route_name": "",
        "num_images": num_images,
        "num_depth": num_depth,
        "num_lidar": num_lidar,
        "num_videos": num_videos,
        "num_nodes": num_nodes,
        "num_objects": num_objects,
        "num_events": num_events,
    }

    manifest_path = os.path.join(output_dir, "dataset_manifest.yaml")
    with open(manifest_path, "w", encoding="utf-8") as f:
        _yaml.dump(manifest, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"✓ 数据集清单: → {manifest_path}")


# ── 主流程 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ScribeMem-Bench 数据集构建主控脚本（改版格式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理所有 session（默认）
  python3 scripts/assemble_dataset.py

  # 指定单个 session
  python3 scripts/assemble_dataset.py -s 001
  python3 scripts/assemble_dataset.py -s 002 --only images,odom

  # 跳过已存在（增量模式）
  python3 scripts/assemble_dataset.py --skip-existing

  # 指定场景目录
  python3 scripts/assemble_dataset.py -s 001 --scene scenes2
        """,
    )
    parser.add_argument("-s", "--session", help="指定 session 编号（如 001），不指定则处理全部")
    parser.add_argument("--bag-dir", default="", help="手动指定 rosbag2 目录（覆盖自动发现）")
    parser.add_argument("--output", default="", help="手动指定输出目录（覆盖自动发现）")
    parser.add_argument("--skip-existing", action="store_true", help="跳过已存在的输出文件")
    parser.add_argument("--only", help="仅运行指定模块 (逗号分隔: odom,tf,images,depth,lidar,nodes)")
    parser.add_argument("--sample-interval", type=float, default=2.0, help="记忆节点时间采样间隔(秒)")
    parser.add_argument("--sample-distance", type=float, default=0.5, help="记忆节点空间采样间隔(米)")
    parser.add_argument("--scene", default="scenes1", help="场景目录名 (默认 scenes1)")
    args = parser.parse_args()

    project_dir = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    raw_dir = os.path.join(project_dir, "raw")
    datasets_dir = os.path.join(project_dir, "datasets", args.scene)

    # 收集要处理的 session
    sessions = []  # list of (label, bag_dir, output_dir)
    if args.bag_dir:
        out = args.output or os.path.join(datasets_dir, os.path.basename(args.bag_dir))
        sessions.append((os.path.basename(args.bag_dir), args.bag_dir, out))
    elif args.session:
        sid = args.session.zfill(3) if args.session.isdigit() else args.session
        sessions.append((f"session_{sid}", os.path.join(raw_dir, f"session_{sid}"),
                         os.path.join(datasets_dir, f"session_{sid}")))
    else:
        raw_globs = sorted(glob.glob(os.path.join(raw_dir, "session_*")))
        for sp in raw_globs:
            sid = os.path.basename(sp)
            if os.path.isdir(sp):
                sessions.append((sid, sp, os.path.join(datasets_dir, sid)))
        if not sessions:
            print(f"错误: 未找到任何 session 目录 (raw/session_*)，请先录制", file=sys.stderr)
            return 1

    print(f"场景: {args.scene}")
    print(f"发现 {len(sessions)} 个 session: {', '.join(s[0] for s in sessions)}")

    global_exit = 0
    for label, bag_dir, output_dir in sessions:
        if _process_one(label, bag_dir, output_dir, args) != 0:
            global_exit = 1

    # 确保 Questions/ 目录存在（QA 占位）
    questions_dir = os.path.join(datasets_dir, "Questions")
    os.makedirs(questions_dir, exist_ok=True)

    return global_exit


def _process_one(label: str, bag_dir: str, output_dir: str, args) -> int:
    """处理单个 session 的提取。"""
    os.makedirs(output_dir, exist_ok=True)
    session_id = label  # e.g. "session_001"

    print(f"\n{'='*60}")
    print(f"  [{label}] {bag_dir}  →  {output_dir}")
    print(f"{'='*60}")

    if not os.path.exists(bag_dir):
        print(f"  ✗ rosbag2 目录不存在: {bag_dir}", file=sys.stderr)
        return 1

    # 确定要运行哪些模块
    all_modules = ["odom", "tf", "images", "depth", "lidar", "nodes"]
    if args.only:
        modules = [m.strip() for m in args.only.split(",")]
        for m in modules:
            if m not in all_modules:
                print(f"错误: 未知模块 '{m}'，可选: {all_modules}", file=sys.stderr)
                return 1
    else:
        modules = all_modules

    # 增量模式检查
    skip_checks = {
        "odom": ["odometry.csv"],
        "tf": ["tf_trajectory.csv", "base_link_pose.csv"],
        "images": ["images_index.csv", "images"],
        "depth": ["depth_index.csv", "depth"],
        "lidar": ["lidar"],
        "nodes": ["memory_nodes.json", "dataset_manifest.yaml"],
        "videos": ["videos", "videos_index.csv"],
    }

    exit_code = 0

    # ---- Phase 1: 里程计 -------------------------------------------------
    if "odom" in modules:
        if args.skip_existing and check_exists(output_dir, skip_checks["odom"]):
            print("\n  ⏭ 跳过里程计 (已存在)")
        else:
            rc = run_script("extract_odom.py", [bag_dir, os.path.join(output_dir, "odometry.csv")], "里程计提取")
            if rc != 0:
                exit_code = rc

    # ---- Phase 2: TF 轨迹 -------------------------------------------------
    if "tf" in modules:
        if args.skip_existing and check_exists(output_dir, skip_checks["tf"]):
            print("\n  ⏭ 跳过 TF (已存在)")
        else:
            rc = run_script("extract_tf.py", [bag_dir, output_dir], "TF 轨迹提取")
            if rc != 0:
                exit_code = rc

    # ---- Phase 3: 图像 ----------------------------------------------------
    if "images" in modules:
        if args.skip_existing and check_exists(output_dir, skip_checks["images"]):
            print("\n  ⏭ 跳过图像 (已存在)")
        else:
            rc = run_script("extract_images.py", [bag_dir, output_dir], "图像提取")
            if rc != 0:
                exit_code = rc

    # ---- Phase 4: 深度图 --------------------------------------------------
    if "depth" in modules:
        if args.skip_existing and check_exists(output_dir, skip_checks["depth"]):
            print("\n  ⏭ 跳过深度图 (已存在)")
        else:
            rc = run_script("extract_depth.py", [bag_dir, output_dir], "深度图提取")
            if rc != 0:
                exit_code = rc

    # ---- Phase 5: 点云 ----------------------------------------------------
    if "lidar" in modules:
        if args.skip_existing and check_exists(output_dir, skip_checks["lidar"]):
            print("\n  ⏭ 跳过点云 (已存在)")
        else:
            rc = run_script("extract_lidar.py", [bag_dir, output_dir], "点云提取")
            if rc != 0:
                exit_code = rc

    # ---- Phase 6: 模板拷贝 ------------------------------------------------
    if "nodes" in modules:
        templates_dir = os.path.join(os.path.abspath(os.path.join(SCRIPT_DIR, "..")), "templates")
        if os.path.isdir(templates_dir):
            for tpl_name in ("camera_calib.yaml", "videos_index.csv", "events.yaml"):
                tpl_src = os.path.join(templates_dir, tpl_name)
                tpl_dst = os.path.join(output_dir, tpl_name)
                if os.path.exists(tpl_src) and not os.path.exists(tpl_dst):
                    shutil.copy2(tpl_src, tpl_dst)
                    # Replace {{session_id}} placeholder in yaml files
                    if tpl_name.endswith(".yaml"):
                        with open(tpl_dst, "r") as f:
                            content = f.read()
                        content = content.replace("{{session_id}}", session_id)
                        with open(tpl_dst, "w") as f:
                            f.write(content)
                    print(f"  ✓ 模板 → {tpl_dst}")

        # 拷贝会话元信息 meta.yaml（从 raw/ 到 datasets/）
        src_meta = os.path.join(bag_dir, "meta.yaml")
        if os.path.exists(src_meta):
            dst_meta = os.path.join(output_dir, "meta.yaml")
            if not os.path.exists(dst_meta):
                shutil.copy2(src_meta, dst_meta)
                print(f"  ✓ 会话元信息 meta.yaml → {dst_meta}")
        else:
            print(f"  ⓘ 未找到 {src_meta}，数据集内无 meta.yaml")

    # ---- Phase 7: 数据增强 ------------------------------------------------
    if "nodes" in modules:
        print(f"\n{'='*60}")
        print(f"  Phase: 数据增强")
        print(f"{'='*60}")
        enrich_images_csv(output_dir)
        enrich_depth_csv(output_dir)

    # ---- Phase 8: 视频编码 ------------------------------------------------
    if "nodes" in modules:
        print(f"\n{'='*60}")
        print(f"  Phase: 视频编码")
        print(f"{'='*60}")
        rc = run_script("build_videos.py", [output_dir], "图像序列 → 视频流")
        if rc != 0:
            print(f"  ⚠ 视频编码失败，继续...")

    # ---- Phase 9: 记忆节点 ------------------------------------------------
    if "nodes" in modules:
        graph_src = os.path.join(bag_dir, "memory", "graph_store.json")
        calib_yaml = os.path.join(output_dir, "camera_calib.yaml")

        if os.path.exists(graph_src):
            rc = run_script(
                "convert_memory_nodes.py",
                [
                    "--graph", graph_src,
                    "--calib", calib_yaml,
                    "--images-csv", os.path.join(output_dir, "images_index.csv"),
                    "--depth-csv", os.path.join(output_dir, "depth_index.csv"),
                    "--odom-csv", os.path.join(output_dir, "odometry.csv"),
                    "--session-id", session_id,
                    "--output", os.path.join(output_dir, "memory_nodes.json"),
                ],
                "记忆节点转换 (graph_store.json → memory_nodes.json)",
            )
            if rc != 0:
                exit_code = rc
        else:
            print(f"\n  ⚠ 未找到 graph_store.json ({graph_src})，回退到轨迹采样方式")
            rc = run_script(
                "build_memory_nodes.py",
                [
                    "--output-dir", output_dir,
                    "--bag-dir", bag_dir,
                    "--sample-interval", str(args.sample_interval),
                    "--sample-distance", str(args.sample_distance),
                ],
                "记忆节点构建 (轨迹采样回退)",
            )
            if rc != 0:
                exit_code = rc

    # ---- Phase 10: 生成 dataset_manifest.yaml -------------------------------
    if "nodes" in modules:
        generate_manifest(output_dir, args.scene, session_id)

    # ---- 汇总 ---------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  输出目录结构:")
    print(f"{'='*60}")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = "  " * (level + 1)
        for f in files[:5]:
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            if size > 1_000_000:
                size_str = f"{size / 1_000_000:.1f} MB"
            elif size > 1_000:
                size_str = f"{size / 1_000:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"{sub_indent}{f} ({size_str})")
        if len(files) > 5:
            print(f"{sub_indent}... 还有 {len(files) - 5} 个文件")

    if exit_code == 0:
        print(f"\n✓ 数据集构建完成! 输出目录: {output_dir}/")
    else:
        print(f"\n⚠ 部分步骤失败 (退出码 {exit_code})", file=sys.stderr)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
