#!/usr/bin/env python3
"""只读：检查腕部相机（彩色+对齐深度）帧率、帧间隔、深度−彩色时间差、深度有效率，并存一张叠加图 ~/wrist_check.jpg。
用法：python3 ~/nero_wrist_check.py [秒数，默认 5]"""
import sys, time, threading, numpy as np, rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 5
rclpy.init(); n = Node("wrist_check"); st = {"color": [], "depth": []}; last = {}
def cb(m, k):
    st[k].append((m.header.stamp.sec * 1e3 + m.header.stamp.nanosec / 1e6, time.time() * 1e3)); last[k] = m
for k, t in (("color", "color/image_raw"), ("depth", "aligned_depth_to_color/image_raw")):
    n.create_subscription(Image, f"/cam/wrist/{t}", lambda m, k=k: cb(m, k), 10)
ex = SingleThreadedExecutor(); ex.add_node(n); threading.Thread(target=ex.spin, daemon=True).start()
time.sleep(DUR); ex.shutdown()
bad = []
for k, v in st.items():
    if len(v) < 2: print(f"{k}: 没有数据"); bad.append(f"{k} 没数据"); continue
    s = np.array([a for a, _ in v]); gap = np.diff(s).max(); lag = np.median([r - a for a, r in v])
    print(f"{k:6s} {len(v) / DUR:5.1f} fps  最大帧间隔 {gap:.0f} ms  收到延迟中位 {lag:.0f} ms")
    if len(v) / DUR < 25: bad.append(f"{k} 帧率低")
    if gap > 100: bad.append(f"{k} 有掉帧（间隔 {gap:.0f} ms）")
if not bad:
    c = np.array([a for a, _ in st["color"]]); d = np.array([a for a, _ in st["depth"]])
    c = c[(c >= d.min() + 50) & (c <= d.max() - 50)]
    dd = np.abs([d[np.argmin(np.abs(d - x))] - x for x in c])
    print(f"深度−彩色 时间差：平均 {np.mean(dd):.2f} ms  最大 {np.max(dd):.2f} ms（要求 ≤ 20）")
    if np.max(dd) > 20: bad.append("深度−彩色时间差超 20 ms")
    m = last["depth"]; dep = np.frombuffer(m.data, np.uint16).reshape(m.height, m.width)
    v = dep[dep > 0]; print(f"深度：有效像素 {(dep > 0).mean() * 100:.0f}%  中位 {np.median(v) if v.size else 0:.0f} mm  中心 {dep[m.height // 2, m.width // 2]} mm")
    if (dep > 0).mean() < 0.5: bad.append("深度有效像素 < 50%（镜头挡住或太近？）")
    try:
        import cv2
        mc = last["color"]; img = np.frombuffer(mc.data, np.uint8).reshape(mc.height, mc.width, -1)[:, :, :3]
        if mc.encoding.lower().startswith("rgb"): img = img[:, :, ::-1]
        col = cv2.applyColorMap(cv2.convertScaleAbs(np.clip(dep, 0, 1000), alpha=255 / 1000), cv2.COLORMAP_JET); col[dep == 0] = 0
        ov = cv2.addWeighted(img, 0.5, cv2.resize(col, (img.shape[1], img.shape[0])), 0.5, 0)
        cv2.imwrite(str(__import__("pathlib").Path.home() / "wrist_check.jpg"), np.hstack([img, ov]))
        print("叠加图已存 ~/wrist_check.jpg（左彩色，右彩色+深度，物体轮廓应对齐）")
    except Exception as e: print("叠加图没存：", e)
print("✓ 腕部相机正常" if not bad else "✗ " + "；".join(bad)); sys.exit(1 if bad else 0)
