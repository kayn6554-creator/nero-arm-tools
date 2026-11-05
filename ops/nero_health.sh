#!/bin/bash
# 一键体检：只读，不动机械臂。看服务器、tmux、各程序、网页端口、CAN、从臂状态、模型服务是否正常。
# 用法：conda deactivate; bash ~/nero_health.sh
source /opt/ros/humble/setup.bash 2>/dev/null
source /data/kaiz/projects/NERO-arm/agx_arm_ws/install/setup.bash 2>/dev/null
G='\033[32m✓\033[0m'; R='\033[31m✗\033[0m'; Y='\033[33m!\033[0m'
echo "== 服务器：$(uptime -p)（$(uptime | sed 's/.*load/load/')）"
if tmux has-session -t nero 2>/dev/null; then
  echo -e "$G tmux 会话 nero：$(tmux list-windows -t nero | wc -l) 个窗口"
  tmux list-windows -t nero -F '     #{window_index}: #{window_name}  （正在运行 #{pane_current_command}）'
else
  echo -e "$R 没有 tmux 会话 nero —— 服务器可能重启过，按“服务器重启”一节从头开机"
fi
echo "== 程序"
chk() {  # 名字  进程匹配  出问题时怎么办
  local pid st
  pid=$(pgrep -f "$2" | head -1)
  if [ -z "$pid" ]; then echo -e "$R $1 没在运行 → $3"; return; fi
  st=$(ps -o stat= -p "$pid" | tr -d ' ')
  if [[ $st == T* ]]; then echo -e "$Y $1 被暂停（Ctrl+Z）→ kill -CONT $pid，再回那个窗口按 Ctrl+C 正常退出后重启"
  else echo -e "$G $1（PID $pid，已运行 $(ps -o etime= -p "$pid" | tr -d ' ')）"; fi
}
chk "从臂驱动"          "agx_arm_ctrl_single"            "窗口 2 重启驱动（看恢复手册 B1）"
chk "头部相机"          "camera_name:=head"              "窗口 3 重启头部相机（B2）"
chk "腕部相机"          "camera_name:=wrist"             "窗口 6：~/nero_cam_wrist.sh（B2）"
chk "软件急停"          "estop_ready.py"                 "窗口 1 重启（B3）"
chk "定位网页 8765"     "app.py --backend ros2"          "窗口 4 重启（B4）"
chk "采集控制台 8766"   "nero_report_server.py"          "窗口 5：~/nero_report_serve.sh（B5）"
chk "模型测试台 8767"   "nero_vla_web.py"                "它的窗口：/usr/bin/python3 ~/nero_vla_web.py（B6）"
chk "汇总页 8777"       "http.server 8777"               "cd /data/panhe/share && python3 -m http.server 8777 --bind 0.0.0.0（B7）"
pgrep -f "python.*nero_record.py" >/dev/null && echo -e "$Y 录制程序正在运行（采集中？不采集就去窗口 0 按 Esc 结束）"
echo "== 网页端口（服务器这边在不在听）"
for p in 8765 8766 8767 8777; do
  if ss -ltn | grep -q ":$p "; then echo -e "$G $p"; else echo -e "$R $p 没有程序在听"; fi
done
echo "== CAN"
for c in can0 can1; do
  s=$(ip -br link show "$c" 2>/dev/null | awk '{print $2}')
  [ "$s" = "UP" ] && echo -e "$G $c UP" || echo -e "$R $c ${s:-不存在}（服务器重启过？看 C2；can1 NO-CARRIER = 主臂没上电）"
done
echo "== 从臂状态"
a=$(timeout 4 ros2 topic echo /feedback/arm_status --once --no-daemon 2>/dev/null | grep -E "^arm_status" | awk '{print $2}')
case "$a" in
  0) echo -e "$G arm_status 0（已使能、正常）";;
  "") echo -e "$R 读不到 /feedback/arm_status（驱动没起来？）";;
  6) echo -e "$Y arm_status 6（没使能 / 刹车抱着）→ 先查限位再使能（C5、C6）";;
  *) echo -e "$Y arm_status $a（不是 0，先别动臂，看恢复手册 B1）";;
esac
[ -f ~/check_limits.py ] && /usr/bin/python3 ~/check_limits.py 2>/dev/null | tail -3
echo "== 模型服务 172.16.11.115:6699"
timeout 3 bash -c "</dev/tcp/172.16.11.115/6699" 2>/dev/null && echo -e "$G 能连上" || echo -e "$R 连不上（学长的模型服务没开，或网络不通）"
echo "== 你的电脑这边：打开网页前，Windows 要开着带 -L 的 ssh 窗口（看恢复手册 A1）"
