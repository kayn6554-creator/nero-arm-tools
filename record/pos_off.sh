#!/bin/bash
# 定位模式关：主臂切回示教，从臂恢复跟随主臂。务必在网页显示 Target reached 并等 2 秒后再执行
source /opt/ros/humble/setup.bash
ros2 service call /nero/positioning std_srvs/srv/SetBool "{data: false}" 2>/dev/null | grep -o "message=.*"
