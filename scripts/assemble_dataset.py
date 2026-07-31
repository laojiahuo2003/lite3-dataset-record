#!/usr/bin/env python3
"""
ScribeMem-Bench 数据集构建主控脚本。

从 rosbag2 中提取全部传感器数据，并组装为结构化的记忆数据集。

用法:
  python3 scripts/assemble_dataset.py                    # 处理所有 session
  python3 scripts/assemble_dataset.py -s 001             # 指定单个
  python3 scripts/assemble_dataset.py -s 002 --skip-existing  # 增量
  python3 scripts/assemble_dataset.py -s 001 --only images,odom

提取流程:
  Phase 1: 里程计提取  extract_odom.py   → odometry.csv
  Phase 2: TF 轨迹提取  extract_tf.py     → base_link_pose.csv + tf_trajectory.csv
  Phase 3: 图像提取    extract_images.py  → images/
  Phase 4: 深度图提取  extract_depth.py   → depth/
  Phase 5: 点云提取    extract_lidar.py   → lidar/
  Phase 6: 记忆节点构建 build_memory_nodes.py → memory_nodes.json
  Phase 7: 会话元信息   meta.yaml          → 从 raw/ 拷入（含 map_id）
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time


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


def main():
    parser = argparse.ArgumentParser(
        description="ScribeMem-Bench 数据集构建主控脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理所有 session（默认）
  python3 scripts/assemble_dataset.py

  # 指定单个 session
  python3 scripts/assemble_dataset.py -s 001
  python3 scripts/assemble_dataset.py -s 002 --only images,odom

  # 跳过已存在（增量模式，所有 session）
  python3 scripts/assemble_dataset.py --skip-existing
        """,
    )
    parser.add_argument("-s", "--session", help="指定 session 编号（如 001），不指定则处理全部")
    parser.add_argument("--bag-dir", default="",
                        help="手动指定 rosbag2 目录（覆盖自动发现）")
    parser.add_argument("--output", default="",
                        help="手动指定输出目录（覆盖自动发现）")
    parser.add_argument("--skip-existing", action="store_true", help="跳过已存在的输出文件")
    parser.add_argument("--only", help="仅运行指定模块 (逗号分隔: odom,tf,images,lidar,nodes)")
    parser.add_argument("--sample-interval", type=float, default=2.0, help="记忆节点时间采样间隔(秒)")
    parser.add_argument("--sample-distance", type=float, default=0.5, help="记忆节点空间采样间隔(米)")
    args = parser.parse_args()

    project_dir = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    raw_dir = os.path.join(project_dir, "raw")
    datasets_dir = os.path.join(project_dir, "datasets")

    # 收集要处理的 session: 优先用 --bag-dir/--output，其次 --session，最后自动发现
    sessions = []  # list of (label, bag_dir, output_dir)
    if args.bag_dir:
        out = args.output or os.path.join(datasets_dir, os.path.basename(args.bag_dir))
        sessions.append((os.path.basename(args.bag_dir), args.bag_dir, out))
    elif args.session:
        sid = args.session.zfill(3) if args.session.isdigit() else args.session
        sessions.append((sid, os.path.join(raw_dir, f"session_{sid}"),
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

    print(f"发现 {len(sessions)} 个 session: {', '.join(s[0] for s in sessions)}")

    global_exit = 0
    for label, bag_dir, output_dir in sessions:
        if _process_one(label, bag_dir, output_dir, args) != 0:
            global_exit = 1

    return global_exit


def _process_one(label: str, bag_dir: str, output_dir: str, args) -> int:
    """处理单个 session 的提取。"""
    os.makedirs(output_dir, exist_ok=True)

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
    }

    exit_code = 0

    # ---- Phase 1: 里程计 ----
    if "odom" in modules:
        if args.skip_existing and check_exists(output_dir, skip_checks["odom"]):
            print("\n  ⏭ 跳过里程计 (已存在)")
        else:
            rc = run_script("extract_odom.py", [bag_dir, os.path.join(output_dir, "odometry.csv")], "里程计提取")
            if rc != 0:
                exit_code = rc

    # ---- Phase 2: TF 轨迹 ----
    if "tf" in modules:
        if args.skip_existing and check_exists(output_dir, skip_checks["tf"]):
            print("\n  ⏭ 跳过 TF (已存在)")
        else:
            rc = run_script("extract_tf.py", [bag_dir, output_dir], "TF 轨迹提取")
            if rc != 0:
                exit_code = rc

    # ---- Phase 3: 图像 ----
    if "images" in modules:
        if args.skip_existing and check_exists(output_dir, skip_checks["images"]):
            print("\n  ⏭ 跳过图像 (已存在)")
        else:
            rc = run_script("extract_images.py", [bag_dir, output_dir], "图像提取")
            if rc != 0:
                exit_code = rc

    # ---- Phase 4: 深度图 ----
    if "depth" in modules:
        if args.skip_existing and check_exists(output_dir, skip_checks["depth"]):
            print("\n  ⏭ 跳过深度图 (已存在)")
        else:
            rc = run_script("extract_depth.py", [bag_dir, output_dir], "深度图提取")
            if rc != 0:
                exit_code = rc

    # ---- Phase 5: 点云 ----
    if "lidar" in modules:
        if args.skip_existing and check_exists(output_dir, skip_checks["lidar"]):
            print("\n  ⏭ 跳过点云 (已存在)")
        else:
            rc = run_script("extract_lidar.py", [bag_dir, output_dir], "点云提取")
            if rc != 0:
                exit_code = rc

    # ---- Phase 6: 记忆节点构建 ----
    if "nodes" in modules:
        rc = run_script(
            "build_memory_nodes.py",
            [
                "--output-dir", output_dir,
                "--bag-dir", bag_dir,
                "--sample-interval", str(args.sample_interval),
                "--sample-distance", str(args.sample_distance),
            ],
            "记忆节点构建",
        )
        if rc != 0:
            exit_code = rc

    # ---- Phase 7: 拷贝会话元信息 meta.yaml 进数据集 ----
    # meta.yaml 是我们的会话元信息（含 map_id 绑定），属于数据集。
    # metadata.yaml 是 rosbag2 自动生成的索引，留在 raw/，不进数据集。
    src_meta = os.path.join(bag_dir, "meta.yaml")
    if os.path.exists(src_meta):
        dst_meta = os.path.join(output_dir, "meta.yaml")
        shutil.copy2(src_meta, dst_meta)
        print(f"\n  ✓ 会话元信息 meta.yaml → {dst_meta}")
    else:
        print(f"\n  ⓘ 未找到 {src_meta}，数据集内无 meta.yaml")

    # ---- 汇总 ----
    print(f"\n{'='*60}")
    print(f"  输出目录结构:")
    print(f"{'='*60}")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = "  " * (level + 1)
        # 只显示前 5 个文件
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
