#!/usr/bin/env bash
# 一键清空 datasets/ 和 raw/ 目录下的所有数据
# 用法: ./bin/clean.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="$PROJECT_DIR/raw"
DATASETS_DIR="$PROJECT_DIR/datasets"

echo "=== 清理数据集产物 ==="

# raw/
if [ -d "$RAW_DIR" ]; then
    count=$(ls "$RAW_DIR" 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        rm -rf "$RAW_DIR"/*
        echo "✓ raw/       已删除 $count 项"
    else
        echo "○ raw/       已为空"
    fi
else
    echo "○ raw/       不存在"
fi

# datasets/
if [ -d "$DATASETS_DIR" ]; then
    count=$(ls "$DATASETS_DIR" 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        rm -rf "$DATASETS_DIR"/*
        echo "✓ datasets/  已删除 $count 项"
    else
        echo "○ datasets/  已为空"
    fi
else
    echo "○ datasets/  不存在"
fi

echo "=== 清理完成 ==="
