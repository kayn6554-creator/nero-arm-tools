#!/bin/bash
# 关闭控制锁：驱动忽略所有话题运动命令（不影响 move_home）
source /opt/ros/humble/setup.bash
ros2 service call /control_enable std_srvs/srv/SetBool "{data: false}"
