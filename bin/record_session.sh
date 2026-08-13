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
# ROS 环境：传输必须用 rmw_zenoh_cpp（与 start.sh 一致）。
# 否则 ros2 bag record 走默认 fastrtps，看不到 zenoh 上的话题，
# 会录出一个 0 消息的空包（session_XXX_0.db3 只有 ~24K）。
# ---------------------------------------------------------------------------
set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash 2>/dev/null || true
set -u
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_zenoh_cpp}"
# 丢弃可能以其他 RMW 启动的残留 daemon，避免 ros2 CLI 连不上 zenoh 图
ros2 daemon stop >/dev/null 2>&1 || true

# Memory service graph location (configured via MEMGRAPH_KEEP_DATA=1 in
# robonix_manifest.yaml so the graph survives across reboots).
MEMORY_GRAPH="${MEMORY_GRAPH:-/home/dog/code/test/robonix/services/memory/memory/graph_store.json}"
MEMORY_IMAGES="${MEMORY_IMAGES:-/home/dog/code/test/robonix/services/memory/data/images}"

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

    # --- RGB 相机内参（出厂标定，保证离线可复现） ---
    /camera/color/camera_info

    # --- 深度图（Orbbec Gemini 330） ---
    /camera/depth/image_raw

    # --- 深度相机内参（已注册到彩色帧） ---
    /camera/depth/camera_info

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
echo "清零 memory service 记忆..."
curl -s -X POST http://127.0.0.1:37798/reset 2>/dev/null \
    && echo "✓ memory service 记忆已清零" \
    || echo "  ⚠ 无法连接 memory service (端口 37798)，跳过清零"
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
lighting_condition: daylight          # daylight / artificial / night
lux_range: ""                         # optional: measured illuminance range
temperature_c: ""                     # optional: ambient temperature
humidity_pct: ""                      # optional: relative humidity
map_id: ""                            # filled by snapshot_ground_truth.py
sensor_configs:
  camera: {model: "Orbbec Gemini 330", resolution: "640x480", fps: 15}
  lidar: {model: "Livox MID-360", range_m: 40}
  imu: {model: "Built-in IMU", rate_hz: 100}
  odometry: {source: "chassis", rate_hz: 20}
  video: {enabled: true, codec: "h264", resolution: "640x480", fps: 15.0, sample_fps: 1.0}
operator_notes: "${NOTES:-}"
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

# ---------------------------------------------------------------------------
# 拷贝 memory service 的 graph_store.json 和图片到 raw session 目录
# 这些是巡检过程中 memory service 自动记录的记忆节点，后续由
# convert_memory_nodes.py 转换为目标 schema
# ---------------------------------------------------------------------------
echo ""
echo "拷贝 memory service 数据..."
if [ -f "$MEMORY_GRAPH" ]; then
    mkdir -p "$SESSION_DIR/memory"
    cp "$MEMORY_GRAPH" "$SESSION_DIR/memory/graph_store.json"
    echo "✓ graph_store.json → $SESSION_DIR/memory/graph_store.json"
else
    echo "  ⚠ 未找到 graph_store.json ($MEMORY_GRAPH)，跳过"
fi
if [ -d "$MEMORY_IMAGES" ]; then
    mkdir -p "$SESSION_DIR/memory/images"
    cp -r "$MEMORY_IMAGES"/* "$SESSION_DIR/memory/images/" 2>/dev/null || true
    img_count=$(find "$SESSION_DIR/memory/images" -type f 2>/dev/null | wc -l)
    echo "✓ memory images ($img_count 文件) → $SESSION_DIR/memory/images/"
else
    echo "  ⓘ 未找到 memory images 目录，跳过"
fi
