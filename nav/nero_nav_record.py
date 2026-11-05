#!/usr/bin/env python3
"""只读：录一次网页导航的全部排查数据，结束后打包成 ~/navlog_<时间>.tgz。
记录：从臂关节角 / 末端位置 / 状态（全速率）、网页 /api/state（每 0.2 s，含 Move 目标）、
头部和腕部相机的彩色 + 对齐深度（每 1 s 一张，Move 期间每 0.2 s）、相机内参、外参和网页配置。
不发任何指令，不订阅 /control/*（网页要求控制话题只有驱动一个订阅者）。
用法：python3 ~/nero_nav_record.py [网页端口，默认 8765]"""
import csv, io, json, os, shutil, sys, tarfile, threading, time, urllib.request
from pathlib import Path
import numpy as np, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import PoseStamped
from PIL import Image as PI

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
WEB = Path("/data/panhe/IK-test-ros")
OUT = Path.home() / time.strftime("navlog_%m%d_%H%M%S"); OUT.mkdir()
for cam in ("head", "wrist"): (OUT / cam).mkdir()
for f in ("arm_config.json", "extrinsics.json", "wrist_extrinsics.json", "camera_config.json"):
    for d in (WEB, Path("/data/kaiz/projects/NERO-arm/IK-test")):
        if (d / f).exists(): shutil.copy(d / f, OUT / f); break

T0 = time.time(); lock = threading.Lock()
arm_rows, web_rows, info, latest, saved = [], [], {}, {}, {"head": [], "wrist": []}
state = {"j": None, "status": None, "moving": False}
now = lambda: time.time() - T0
stamp = lambda m: m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 - T0

rclpy.init(); n = Node("nero_nav_record")
def on_j(m):
    d = dict(zip(m.name, m.position))
    if all(f"joint{i}" in d for i in range(1, 8)): state["j"] = [d[f"joint{i}"] for i in range(1, 8)]
def on_tcp(m):
    if state["j"] is None: return
    p, q = m.pose.position, m.pose.orientation
    with lock: arm_rows.append([round(now(), 4), p.x, p.y, p.z, q.x, q.y, q.z, q.w] + state["j"] + [state["status"]])
def on_status(m):
    state["status"] = int(m.arm_status)
n.create_subscription(JointState, "/feedback/joint_states", on_j, qos_profile_sensor_data)
n.create_subscription(PoseStamped, "/feedback/tcp_pose", on_tcp, qos_profile_sensor_data)
try:
    from agx_arm_msgs.msg import AgxArmStatus
    n.create_subscription(AgxArmStatus, "/feedback/arm_status", on_status, qos_profile_sensor_data)
except Exception as e:
    print("没有 agx_arm_msgs（没 source agx_arm_ws），不记录 arm_status：", e)
rel = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
for cam in ("head", "wrist"):
    for k, t in (("color", "color/image_raw"), ("depth", "aligned_depth_to_color/image_raw")):
        n.create_subscription(Image, f"/cam/{cam}/{t}", lambda m, c=cam, k=k: latest.__setitem__((c, k), m), rel)
    n.create_subscription(CameraInfo, f"/cam/{cam}/color/camera_info",
                          lambda m, c=cam: info.setdefault(c, dict(width=m.width, height=m.height, k=list(m.k), d=list(m.d), model=m.distortion_model)), 10)
threading.Thread(target=rclpy.spin, args=(n,), daemon=True).start()

def poll_web():
    while not stop.is_set():
        try:
            s = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/state", timeout=1).read())
            state["moving"] = (s.get("motion") or {}).get("state") == "moving"
            with lock: web_rows.append(dict(t=round(now(), 3), state=s))
        except Exception as e:
            with lock: web_rows.append(dict(t=round(now(), 3), error=str(e)))
        time.sleep(0.2)

def save_cams():
    last = 0
    while not stop.is_set():
        gap = 0.2 if state["moving"] else 1.0
        if now() - last < gap: time.sleep(0.02); continue
        last = now()
        for cam in ("head", "wrist"):
            c, d = latest.get((cam, "color")), latest.get((cam, "depth"))
            if c is None or d is None: continue
            k = len(saved[cam])
            rgb = np.frombuffer(c.data, np.uint8).reshape(c.height, c.step)[:, :c.width * 3].reshape(c.height, c.width, 3)
            if c.encoding == "bgr8": rgb = rgb[:, :, ::-1]
            PI.fromarray(np.ascontiguousarray(rgb)).save(OUT / cam / f"{k:04d}_color.jpg", quality=90)
            dep = np.frombuffer(d.data, np.uint16).reshape(d.height, d.step // 2)[:, :d.width]
            np.savez_compressed(OUT / cam / f"{k:04d}_depth.npz", depth=dep)
            saved[cam].append(dict(i=k, t=round(last, 3), color_stamp=round(stamp(c), 4), depth_stamp=round(stamp(d), 4),
                                   moving=state["moving"]))

stop = threading.Event()
threading.Thread(target=poll_web, daemon=True).start(); threading.Thread(target=save_cams, daemon=True).start()
print(f"记录中 → {OUT}\n1) 等 3 秒  2) 网页点物体，把 SELECTED TARGET 那段 JSON 复制下来  3) 按 Move（手放急停）\n4) 停下后等 3 秒，回这里按 Ctrl+C", flush=True)
try:
    while True:
        time.sleep(2)
        print(f"  {now():5.0f}s  臂反馈 {len(arm_rows)} 帧  网页 {len(web_rows)} 次  头部 {len(saved['head'])} 张  腕部 {len(saved['wrist'])} 张"
              f"{'  [移动中]' if state['moving'] else ''}", flush=True)
except KeyboardInterrupt:
    pass
stop.set(); time.sleep(0.5)
with lock:
    with open(OUT / "arm.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t", "x", "y", "z", "qx", "qy", "qz", "qw"] + [f"j{i}" for i in range(1, 8)] + ["arm_status"]); w.writerows(arm_rows)
    (OUT / "web_state.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in web_rows))
(OUT / "meta.json").write_text(json.dumps(dict(start=time.strftime("%F %T", time.localtime(T0)), camera_info=info, frames=saved,
                                               note="t 都是相对 start 的秒；深度 npz 里是 16 位原始值（ROS 16UC1，毫米）"), ensure_ascii=False, indent=1))
sel = input("\n把网页 SELECTED TARGET 的 JSON 粘贴到这里（粘完按回车，再按 Ctrl+D；没有就直接 Ctrl+D）：\n") if sys.stdin.isatty() else ""
try:
    rest = sys.stdin.read()
except Exception:
    rest = ""
(OUT / "selection.txt").write_text(sel + "\n" + rest)
tgz = OUT.with_suffix(".tgz")
with tarfile.open(tgz, "w:gz") as t: t.add(OUT, arcname=OUT.name)
moves = [r for r in web_rows if (r.get("state") or {}).get("motion", {}).get("state") == "moving"]
print(f"\n完成：{tgz}（{tgz.stat().st_size / 1e6:.1f} MB）  臂反馈 {len(arm_rows)} 帧，头部 {len(saved['head'])} 张，腕部 {len(saved['wrist'])} 张，"
      f"网页记录到移动 {len(moves)} 次")
