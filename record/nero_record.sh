#!/bin/bash
# 用法：~/nero_record.sh 数据集名 "英文任务描述"
# 例子：~/nero_record.sh nero_pick_v1 "pick up the red cube and place it in the box"
# 循环录制，只靠按键推进；结束在复位阶段按 Esc。目录已存在则自动接着录（--resume）。
# 也可以用任务预设（~/nero_tasks.json）：~/nero_record.sh weigh     → 数据集、任务描述都从预设读
if [ $# -ge 1 ] && [ $# -lt 2 ] || { [ $# -ge 1 ] && [[ "$2" == --* ]]; }; then
  PRESET=$(python3 -c "import json,sys; t=json.load(open('$HOME/nero_tasks.json')).get('$1'); print(t['dataset']+'\t'+t['task']) if t and t.get('task') else sys.exit(1)") \
    || { echo "没有这个任务预设：$1（看 ~/nero_tasks.json）"; exit 1; }
  NAME=$(echo "$PRESET" | cut -f1); TASK=$(echo "$PRESET" | cut -f2); shift 1
elif [ $# -ge 2 ]; then NAME=$1; TASK=$2; shift 2
else echo "用法：$0 任务预设（如 weigh）  或  $0 数据集名 \"任务描述\""; exit 1; fi
export NERO_DATASET=$NAME NERO_DATA_ROOT=/data/panhe/lerobot_data
source ~/miniconda3/etc/profile.d/conda.sh
source ~/env_evo.sh
ROOT=/data/panhe/lerobot_data/$NAME
RESUME=false
if [ -f "$ROOT/meta/info.json" ]; then
  if ls "$ROOT"/meta/episodes/chunk-*/*.parquet >/dev/null 2>&1; then RESUME=true
  else BAD=$ROOT.empty_$(date +%H%M%S); mv "$ROOT" "$BAD"; echo "上次没存下任何一条，已移到 $BAD，重新新建"; fi
fi
echo "数据集：$ROOT   接着录：$RESUME   任务：$TASK"
echo; echo "== 开录前检查（1/3 从臂  2/3 相机  3/3 主从对齐）=="
ST=$(timeout 4 ros2 topic echo /feedback/arm_status --once --no-daemon 2>/dev/null | awk '/^arm_status/{print $2}')
if [ "$ST" != "0" ]; then echo "✗ 1/3 从臂未就绪（arm_status=${ST:-读不到}）：驱动在运行吗？断电后按看板〇节查限位、使能"; exit 1; fi
echo "✓ 1/3 从臂已使能"
OUT=$(python ~/nero_cam_check.py 3 2>&1); RC=$?; echo "$OUT" | tail -3
[ $RC -eq 0 ] || { echo "✗ 2/3 相机检查没过"; exit 1; }
echo "✓ 2/3 相机正常"
python ~/nero_init.py || { echo "✗ 3/3 主从对齐没完成"; exit 1; }
echo "✓ 3/3 主从已对齐"; echo
python ~/nero_record.py \
  --robot.type=nero_follower --robot.id=nero --robot.read_only=false \
  --robot.cameras='{ wrist: {type: ros_image, topic: "/cam/wrist/color/image_raw", depth_topic: "/cam/wrist/aligned_depth_to_color/image_raw", width: 640, height: 480, fps: 30}, front: {type: ros_image, topic: "/cam/head/color/image_raw", depth_topic: "/cam/head/aligned_depth_to_color/image_raw", width: 640, height: 480, fps: 15}}' \
  --robot.camera_sync_ref=front \
  --teleop.type=nero_leader --teleop.id=nero_leader \
  --dataset.repo_id=panhe/$NAME \
  --dataset.root=$ROOT \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes=1000 \
  --dataset.episode_time_s=3600 \
  --dataset.reset_time_s=3600 \
  --dataset.push_to_hub=false \
  --display_data=false \
  --play_sounds=false \
  --resume=$RESUME "$@"

python ~/nero_report.py "$ROOT"
