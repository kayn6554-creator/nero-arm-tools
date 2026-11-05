#!/usr/bin/env python3
# 只读：从臂当前关节角 vs SDK 限位（j7 收紧到 ±1.39）
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from pyAgxArm.api.constants import ROBOT_JOINT_LIMIT_PRESET_RAD as L

lims = list(L['nero'].values())
lims[6] = (-1.39, 1.39)
names = [f"joint{i}" for i in range(1, 8)]
got = {}

rclpy.init()
n = Node('check_limits')
n.create_subscription(JointState, '/feedback/joint_states',
                      lambda m: got.update(zip(m.name, m.position)), 10)
for _ in range(50):
    rclpy.spin_once(n, timeout_sec=0.1)
    if all(k in got for k in names):
        break
if not all(k in got for k in names):
    raise SystemExit("没读到关节角，驱动在运行吗？")

bad = 0
for k, (lo, hi) in zip(names, lims):
    v = got[k]
    ok = lo <= v <= hi
    bad += not ok
    print(f"{k}: {v:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {'OK' if ok else '超限!'}")
if 'gripper' in got:
    print(f"gripper: {got['gripper']:.4f} m")
print("全部在限位内" if bad == 0 else f"{bad} 个关节超限：先别动，贴给我")
n.destroy_node()
rclpy.shutdown()
