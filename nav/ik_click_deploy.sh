#!/bin/bash
# 把学长的 Click-to-Reach（/data/kaiz/projects/NERO-arm/IK-test）完整复制一份到 /data/panhe/ik_click，之后只改这份。
# 复制的是学长目录当前的样子（含没提交的改动）；配置文件是真复制，不再是软链接。
# Python 环境（.venv）和 ROS 接口（ros_ws）体积大、只读用，直接用学长的路径，不复制。
# 再盖上我们的三处改动：ROS 头部相机（ros_camera.py + app.py 一行）、导航安全检查（nav_guard.py + 两行调用）、悬停高度 20 cm。
set -e
SRC=/data/kaiz/projects/NERO-arm/IK-test
DST=/data/panhe/ik_click
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -e "$DST" ]; then mv "$DST" "${DST}_old_$(date +%m%d_%H%M%S)"; echo "旧的 $DST 已改名备份"; fi
mkdir -p "$DST"
rsync -a --exclude .venv --exclude ros_ws --exclude __pycache__ --exclude '*.log' --exclude '*.pid' "$SRC/" "$DST/"
cd "$DST"
echo "学长代码版本：$(git -c safe.directory='*' log -1 --format='%h %ad %s' --date=short 2>/dev/null || echo 无 git)"
git -c safe.directory='*' status --short 2>/dev/null | head -20 | sed 's/^/  学长没提交的改动：/' || true
# 我们的 app.py / ros_robot.py 是在 10-28 的学长版本上改的；学长之后改过的话先停下，不覆盖
echo "4670015a347919d819fa7a7114a7372b  app.py
55cea722d55a225fe903cf2d6f433ca0  ros_robot.py" | md5sum -c --quiet || { echo "学长的 app.py 或 ros_robot.py 和 10-28 不一样了，没有覆盖。把这条输出发给我"; exit 1; }
cp "$HERE"/{ros_camera.py,nav_guard.py,ros_robot.py,app.py} "$DST/"
python3 - <<'PY'
import json; p='arm_config.json'; c=json.load(open(p)); old=c['approach_offset_base_m']; c['approach_offset_base_m']=[0, 0, 0.2]
json.dump(c, open(p, 'w'), indent=2); print('悬停高度', old, '→', c['approach_offset_base_m'])
PY
sed -i "s#source ../agx_arm_ws/install/setup.bash#source /data/kaiz/projects/NERO-arm/agx_arm_ws/install/setup.bash#; s#source ros_ws/install/setup.bash#source $SRC/ros_ws/install/setup.bash#; s#\.venv/bin/python#$SRC/.venv/bin/python#g" start_ros_app.sh
chmod +x start_ros_app.sh
grep -n "source\|python" start_ros_app.sh | head
$SRC/.venv/bin/python -c "import app, nav_guard, ros_camera; print('部署好了：$DST')"
