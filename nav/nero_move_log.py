#!/usr/bin/env python3
"""只读：记录网页 Move 时从臂实际怎么走（关节角 + 末端位置），看它是不是先往下冲。
开着它，然后在网页按 Move；到位或按 Stop 后回到这里按 Ctrl+C，会打印分析并存 CSV。
用法：python3 ~/nero_move_log.py"""
import csv, math, signal, sys, time, threading
import numpy as np, rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
PLANE = (0.0157, -0.0297, -0.0423)          # 台面高度 z = a + b·x + c·y（10-28 标定）
tz = lambda x, y: PLANE[0] + PLANE[1] * x + PLANE[2] * y
rows, cmd = [], {}
rclpy.init(); n = Node("nero_move_log"); last_j = [None]
def on_j(m):
    d = dict(zip(m.name, m.position))
    if all(f"joint{i}" in d for i in range(1, 8)): last_j[0] = [d[f"joint{i}"] for i in range(1, 8)]
def on_t(m):
    if last_j[0] is None: return
    p = m.pose.position; rows.append([time.time(), p.x, p.y, p.z] + last_j[0])
n.create_subscription(JointState, "/feedback/joint_states", on_j, qos_profile_sensor_data)
n.create_subscription(PoseStamped, "/feedback/tcp_pose", on_t, qos_profile_sensor_data)
# 不订阅 /control/move_j：网页要求这个话题只有驱动一个订阅者，多一个它会拒绝 Move
threading.Thread(target=rclpy.spin, args=(n,), daemon=True).start()
print("记录中……去网页按 Move，到位/停下后在这里按 Ctrl+C", flush=True)
try:
    while True: time.sleep(0.2)
except KeyboardInterrupt:
    pass
if not rows: sys.exit("没收到反馈（从臂驱动在跑吗？）")
fn = time.strftime("/home/panhe/move_log_%m%d_%H%M%S.csv")
with open(fn, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["t", "x", "y", "z"] + [f"j{i}" for i in range(1, 8)]); w.writerows(rows)
R = np.array(rows)
mv = np.abs(np.diff(R[:, 4:], axis=0)).max(1) > 1e-3          # 关节开始动的时刻当 0 s
k0 = max(0, int(np.argmax(mv)) - 5) if mv.any() else 0
R = R[k0:]; t = R[:, 0] - R[5 if len(R) > 5 else 0, 0]
cmd["q"] = R[-1, 4:]                                           # 终点关节角当目标
h = R[:, 3] - tz(R[:, 1], R[:, 2])
print(f"\n存了 {len(R)} 帧 → {fn}")
print(f"末端离台面：起点 {h[0]*100:.1f} cm，终点 {h[-1]*100:.1f} cm，最低 {h.min()*100:.1f} cm（第 {t[h.argmin()]:.1f} s，xy=({R[h.argmin(),1]:.2f},{R[h.argmin(),2]:.2f})）")
if "q" in cmd:
    q0, q1 = R[0, 4:], np.array(cmd["q"])
    for i in range(7):
        dq = q1[i] - q0[i]
        if abs(dq) < 0.05: continue
        prog = (R[:, 4 + i] - q0[i]) / dq
        def when(p):
            k = np.argmax(prog >= p); return t[k] if prog[k] >= p else float("nan")
        print(f"  关节{i+1} 转 {math.degrees(dq):+6.0f}°：走到 50% 用 {when(.5):4.1f} s，90% 用 {when(.9):4.1f} s")
print("每 0.5 s 一行：时间 / 末端 xyz / 离台面")
nxt = -1
for k in range(len(R)):
    if t[k] >= nxt:
        print(f"  {t[k]:5.1f}s  ({R[k,1]:+.2f},{R[k,2]:+.2f},{R[k,3]:+.2f})  {h[k]*100:5.1f} cm"); nxt = t[k] + 0.5
