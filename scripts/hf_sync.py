#!/usr/bin/env python3
"""Hugging Face 数据集上传/下载同步脚本 (ScribeMem-Bench)

国内网络直连 huggingface.co 不通，默认走官方镜像 hf-mirror.com（上传/下载都支持）。

用法:
  # 上传所有地图数据
  python3 scripts/hf_sync.py upload YOUR_USERNAME/scribemem-bench

  # 上传指定地图
  python3 scripts/hf_sync.py upload YOUR_USERNAME/scribemem-bench --map map_f1

  # 连同 raw rosbag 一起传
  python3 scripts/hf_sync.py upload YOUR_USERNAME/scribemem-bench --include raw --sessions 001,002

  # 下载（到本地目录）
  python3 scripts/hf_sync.py download YOUR_USERNAME/scribemem-bench --local-dir ./hf_data

环境变量:
  HF_ENDPOINT  镜像地址，默认 https://hf-mirror.com
  HF_TOKEN     你的写权限 token；或先执行 `hf auth login`（推荐）

首次使用（只需要一次）:
  1. 去 https://huggingface.co/settings/tokens 创建 Read+Write token（浏览器可能需代理）
  2. export HF_ENDPOINT=https://hf-mirror.com
     hf auth login     # 粘贴 token
  3. 创建数据集仓库（或用 HF 网页 Create → Dataset）:
     hf repo create scribemem-bench --type=dataset
"""
import argparse
import os
import sys

from huggingface_hub import HfApi

# 默认直连。若直连不可用（网络波动），可临时 export HF_ENDPOINT=https://hf-mirror.com
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

api = HfApi(endpoint=ENDPOINT)


def _ensure_login() -> None:
    try:
        info = api.whoami()
        print(f"✓ 已登录: {info.get('name', '?')}")
    except Exception as e:
        sys.exit(
            "✗ 未登录。请先执行:\n"
            f"    export HF_ENDPOINT={ENDPOINT}\n"
            "    hf auth login\n"
            f"（token 在 https://huggingface.co/settings/tokens 创建，需 Read+Write 权限）\n"
            f"原始错误: {e}"
        )


def upload(args) -> None:
    _ensure_login()

    datasets_root = os.path.join(PROJECT_DIR, "datasets")

    # 新结构：支持上传单个地图或所有地图
    if args.map:
        map_dir = os.path.join(datasets_root, args.map)
        if not os.path.isdir(map_dir):
            sys.exit(f"✗ 目录不存在: {map_dir}")

        print(f"▸ 上传到 {args.repo_id} (端点 {ENDPOINT})")
        print(f"  来源: {map_dir}/")
        print(f"  远端: {args.map}/")
        print("  已上传的文件会自动跳过（增量、断点续传）...\n")

        api.upload_large_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=datasets_root,
            allow_patterns=[f"{args.map}/**"],
        )
    else:
        # 上传所有 map_* 目录
        maps = [d for d in os.listdir(datasets_root)
                if os.path.isdir(os.path.join(datasets_root, d)) and d.startswith("map_")]

        if not maps:
            sys.exit(f"✗ 未找到任何 map_* 目录在 {datasets_root}")

        print(f"▸ 上传到 {args.repo_id} (端点 {ENDPOINT})")
        print(f"  来源: {datasets_root}/")
        print(f"  发现 {len(maps)} 个地图: {', '.join(maps)}")
        print("  已上传的文件会自动跳过（增量、断点续传）...\n")

        for map_name in maps:
            print(f"  ▸ 上传 {map_name}/ ...")
            api.upload_large_folder(
                repo_id=args.repo_id,
                repo_type="dataset",
                folder_path=datasets_root,
                allow_patterns=[f"{map_name}/**"],
            )

    if args.include == "raw" and args.sessions:
        raw_root = os.path.join(PROJECT_DIR, "raw")
        for s in args.sessions.split(","):
            s = s.strip()
            if not os.path.isdir(os.path.join(raw_root, f"session_{s}")):
                print(f"  ⚠ 跳过不存在的 raw/session_{s}")
                continue
            print(f"  ▸ 上传 raw/session_{s} ...")
            api.upload_large_folder(
                repo_id=args.repo_id,
                repo_type="dataset",
                folder_path=raw_root,
                allow_patterns=[f"session_{s}/**"],
            )
    print("\n✓ 上传完成")


def download(args) -> None:
    from huggingface_hub import snapshot_download

    _ensure_login()
    local = args.local_dir or os.path.join(PROJECT_DIR, "hf_data")
    print(f"▸ 从 {args.repo_id} 下载到 {local} (端点 {ENDPOINT})")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=local,
        max_workers=4,
    )
    print(f"\n✓ 下载完成 → {local}")


def delete(args) -> None:
    """删除远程仓库中的文件或目录"""
    _ensure_login()

    path = args.path.strip("/")
    if not path:
        sys.exit("✗ 路径不能为空")

    print(f"▸ 从 {args.repo_id} 删除: {path}")

    if not args.yes:
        confirm = input(f"  确认删除 '{path}' ? [y/N] ")
        if confirm.lower() != 'y':
            print("✗ 已取消")
            return

    try:
        api.delete_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            path_in_repo=path,
        )
        print(f"✓ 已删除: {path}")
    except Exception as e:
        sys.exit(f"✗ 删除失败: {e}")


def main():
    p = argparse.ArgumentParser(description="HF 数据集同步 (ScribeMem-Bench)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pu = sub.add_parser("upload", help="上传本地数据集到 HF")
    pu.add_argument("repo_id", help="如 YOUR_USERNAME/scribemem-bench")
    pu.add_argument("--map", default="", help="指定地图目录名（如 map_f1），留空则上传所有 map_*")
    pu.add_argument("--include", choices=["datasets", "raw"], default="datasets",
                    help="只传 datasets，或连同 raw rosbag 一起传")
    pu.add_argument("--sessions", default="", help="要一起上传的 raw session，如 001,002")
    pu.set_defaults(func=upload)

    pd = sub.add_parser("download", help="从 HF 下载数据集到本地")
    pd.add_argument("repo_id", help="如 YOUR_USERNAME/scribemem-bench")
    pd.add_argument("--local-dir", default="", help="本地目标目录 (默认 ./hf_data)")
    pd.set_defaults(func=download)

    pdel = sub.add_parser("delete", help="从 HF 仓库删除文件或目录")
    pdel.add_argument("repo_id", help="如 YOUR_USERNAME/scribemem-bench")
    pdel.add_argument("path", help="要删除的路径（如 scenes1）")
    pdel.add_argument("-y", "--yes", action="store_true", help="跳过确认")
    pdel.set_defaults(func=delete)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
