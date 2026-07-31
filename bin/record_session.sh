#!/usr/bin/env bash
# ============================================================================
# ScribeMem-Bench 巡检录制脚本
#
# 用法:
#   ./bin/record_session.sh                    # 自动编号，使用默认话题
#   ./bin/record_session.sh 003                # 指定 session 编号
#   ./bin/record_session.sh 003 "夜间巡检"     # 指定编号 + 备注
#
# 输出:
#   raw/session_XXX/                           # rosbag2 目录
#   raw/session_XXX/meta.yaml                  # 自动生成的元信息模板
# ============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="$PROJECT_DIR/raw"
SESSION_NUM="${1:-}"
NOTES="${2:-}"

# ---------------------------------------------------------------------------
# 自动检测下一个 session 编号
# ---------------------------------------------------------------------------
if [ -z "$SESSION_NUM" ]; then
    LAST_NUM=$(ls -d "$RAW_DIR"/session_* 2>/dev/null | sed 's/.*session_//' | sort -n | tail -1 || true)
    if [ -z "$LAST_NUM" ]; then
        SESSION_NUM="001"
    else
        SESSION_NUM=$(printf "%03d" $((10#$LAST_NUM + 1)))
    fi
fi

SESSION_DIR="$RAW_DIR/session_$SESSION_NUM"

if [ -d "$SESSION_DIR" ]; then
    echo "⚠ Session 目录已存在: $SESSION_DIR"
    echo "  如需覆盖请先删除: rm -rf $SESSION_DIR"
    exit 1
fi

# ---------------------------------------------------------------------------
# 话题列表（与 ScribeMem-Bench 数据集要求对齐）
# ---------------------------------------------------------------------------
# 核心话题（必需）
TOPICS=(
    # --- RGB 图像 ---
    /camera/color/image_raw

    # --- 深度图（Orbbec Gemini 330） ---
    /camera/depth/image_raw

    # --- 激光雷达点云 ---
    /scanner/cloud

    # --- 里程计 ---
    /odom

    # --- TF 坐标变换 ---
    /tf
    /tf_static

    # --- IMU（可选，用于姿态校验） ---
    /imu/data
)

echo "============================================================================"
echo "  ScribeMem-Bench 巡检录制"
echo "============================================================================"
echo "  Session:    session_$SESSION_NUM"
echo "  输出目录:   $SESSION_DIR"
echo "  录制话题:   ${#TOPICS[@]} 个"
echo ""
for topic in "${TOPICS[@]}"; do
    echo "    - $topic"
done
echo ""
echo "  备注:       ${NOTES:-（无）}"
echo "============================================================================"
echo ""
echo "按 Ctrl-C 停止录制"
echo ""

# ---------------------------------------------------------------------------
# 开始录制
# ---------------------------------------------------------------------------
ros2 bag record -o "$SESSION_DIR" "${TOPICS[@]}"

# ---------------------------------------------------------------------------
# 录制结束后生成 meta.yaml 模板
# ---------------------------------------------------------------------------
cat > "$SESSION_DIR/meta.yaml" << EOF
session_id: "$SESSION_NUM"
timestamp: "$(date '+%Y-%m-%d %H:%M:%S')"
lighting: daylight          # daylight / artificial / night
notes: "${NOTES:-}"
EOF

echo ""
echo "✓ 录制完成 → $SESSION_DIR"
echo "✓ meta.yaml 已生成"

# ---------------------------------------------------------------------------
# 趁服务还活着，自动抓一次 Ground Truth 快照（物体 + 场景/平面图）
# 服务没起来时不会中断流程，失败仅提示
# ---------------------------------------------------------------------------
echo ""
echo "抓取 Ground Truth 快照（scene 物体 + 场景定义）..."
python3 "$PROJECT_DIR/scripts/snapshot_ground_truth.py" "$SESSION_NUM" \
    || echo "  ⚠ GT 快照未生成（scene/mapping 服务可能未运行），可稍后手动补跑:
      python3 scripts/snapshot_ground_truth.py $SESSION_NUM"
