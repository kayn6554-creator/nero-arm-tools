#!/usr/bin/env python3
"""只读：按录制程序的方式（每个相机一个节点一个线程、可靠模式）收两相机彩色+深度，看帧率、帧间隔和配对时间差。"""
import sys, time, threading, numpy as np, rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image
DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 5
rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
st = {}; exs = []
for cam in ("head", "wrist"):
    n = Node(f"check_{cam}")
    for k, t in (("color", "color/image_raw"), ("depth", "aligned_depth_to_color/image_raw")):
        st[f"{cam}/{k}"] = []
        n.create_subscription(Image, f"/cam/{cam}/{t}", lambda m, k=f"{cam}/{k}": st[k].append((m.header.stamp.sec * 1e3 + m.header.stamp.nanosec / 1e6, time.time() * 1e3)), 10)
    ex = SingleThreadedExecutor(); ex.add_node(n); threading.Thread(target=ex.spin, daemon=True).start(); exs.append(ex)
time.sleep(DUR)
for ex in exs: ex.shutdown()
for k, v in st.items():
    gap = np.diff([s for s, _ in v]).max() if len(v) > 1 else float("nan")
    print(f"{k:12s} {len(v) / DUR:5.1f} fps   最大帧间隔 {gap:.0f} ms")
dead = [k for k, v in st.items() if len(v) < 2]
if dead: sys.exit(f"✗ 这些相机话题没有数据：{', '.join(dead)}（窗口 3 头部 / 窗口 6 腕部 驱动在运行吗？）")
h = np.array([s for s, _ in st["head/color"]]); w = np.array([s for s, _ in st["wrist/color"]])
h = h[(h >= w.min() + 50) & (h <= w.max() - 50)]   # 只看两边都在收的时间段（刚订阅时有几百 ms 的发现延迟）
d = np.abs(np.array([w[np.argmin(np.abs(w - x))] - x for x in h]))
print(f"配对时间差：平均 {d.mean():.1f} ms，95% {np.percentile(d, 95):.1f} ms，最大 {d.max():.1f} ms，超过 20 ms 的 {int((d > 20).sum())}/{len(d)} 帧")
hc = np.array([s for s, _ in st["head/color"]]); hd = np.array([s for s, _ in st["head/depth"]])
hc = hc[(hc >= hd.min() + 50) & (hc <= hd.max() - 50)]
dd = np.abs(np.array([hd[np.argmin(np.abs(hd - x))] - x for x in hc]))
print(f"头部 深度−彩色 时间差：平均 {dd.mean():.2f} ms，最大 {dd.max():.2f} ms（要求 ≤ 20）")
lag = [r - s for s, r in st["head/color"] + st["wrist/color"]]
print(f"时间戳到收到的延迟：中位 {np.median(lag):.0f} ms")
if d.max() > 20: sys.exit("✗ 两相机配对时间差超过 20 ms")
if dd.max() > 20: sys.exit("✗ 头部深度和彩色时间差超过 20 ms")
