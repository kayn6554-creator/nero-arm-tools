#!/bin/bash
# 腕部相机 ROS 驱动（tmux 窗口 6）。和头部一样带对齐深度和系统时间戳。
source /opt/ros/humble/setup.bash
ros2 launch realsense2_camera rs_launch.py serial_no:=_262422272398 \
  camera_namespace:=cam camera_name:=wrist initial_reset:=true \
  depth_module.color_profile:=640,480,30 depth_module.depth_profile:=640,480,30 \
  align_depth.enable:=true
