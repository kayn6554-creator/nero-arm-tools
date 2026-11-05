#!/bin/bash
# 定位模式开：从臂不再执行主臂指令，主臂跟随从臂（网页可以移动从臂）
source /opt/ros/humble/setup.bash
ros2 service call /nero/positioning std_srvs/srv/SetBool "{data: true}" 2>/dev/null | grep -o "message=.*"
