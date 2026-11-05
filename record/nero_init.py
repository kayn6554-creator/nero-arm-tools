#!/usr/bin/env python3
"""初始化：主臂转到从臂姿态（从臂不动），然后切回示教。录制程序没在运行时使用。
  python ~/nero_init.py        对齐（默认）
  （--save / --origin 为保存、回到固定原位的备用功能，一般不用）
"""
import json, math, sys, threading, time
from pathlib import Path
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool
from lerobot.teleoperators.utils import make_teleoperator_from_config
from lerobot_teleoperator_nero import NeroLeaderConfig, nero_joint_limits

ORIGIN = Path.home() / "nero_origin.json"
MODE = "save" if "--save" in sys.argv else "origin" if "--origin" in sys.argv else "align"
names = [f"joint{i}" for i in range(1, 8)]; keys = [f"{n}.pos" for n in names]
LIM = nero_joint_limits(); deg = math.degrees

rclpy.init(signal_handler_options=SignalHandlerOptions.NO); node = Node("panhe_nero_init")
fol, fol_t = {}, [0.0]
def cb(m): fol.update(zip(m.name, m.position)); fol_t[0] = time.monotonic()
node.create_subscription(JointState, "/feedback/joint_states", cb, 10)
pub = node.create_publisher(JointState, "/control/move_j", 1)
gate = node.create_client(SetBool, "/control_enable")
ex = SingleThreadedExecutor(); ex.add_node(node); th = threading.Thread(target=ex.spin, daemon=True); th.start()

def fq(): return [fol[n] for n in names]
def set_gate(v):
    if not gate.wait_for_service(timeout_sec=3): print("控制锁服务无响应"); return
    r = SetBool.Request(); r.data = v; f = gate.call_async(r); t0 = time.time()
    while not f.done() and time.time() - t0 < 3: time.sleep(0.02)
    print("  控制锁:", f.result().message if f.done() and f.result() else "无响应")
def over_limit(q): return [f"j{i+1} {deg(v):.1f}°" for i, v in enumerate(q) if not LIM[i][0] <= v <= LIM[i][1]]
def wait_until(cond, timeout, msg):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond(): return True
        time.sleep(0.05)
    print(f"  超时：{msg}"); return False

t0 = time.time()
while not all(n in fol for n in names) and time.time() - t0 < 3: time.sleep(0.05)
if not all(n in fol for n in names): raise SystemExit("读不到从臂关节：驱动在运行吗？")
q = fq(); bad = over_limit(q)
print("从臂当前:", " ".join(f"{deg(v):6.1f}" for v in q))
if bad: raise SystemExit(f"从臂有关节超限（{', '.join(bad)}）：先把主臂掰进限位，再运行 align_follower.py。")

if MODE == "save":
    ORIGIN.write_text(json.dumps({"joints": q, "saved": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    print(f"已把从臂当前姿态存为原位：{ORIGIN}"); ex.shutdown(); th.join(2); node.destroy_node(); rclpy.shutdown(); sys.exit(0)

origin = None
if MODE == "origin":
    if not ORIGIN.exists(): raise SystemExit("还没有原位：先用 --align 对齐，遥操作到想要的位置，再 --save。")
    origin = json.loads(ORIGIN.read_text())["joints"]
    if over_limit(origin): raise SystemExit("保存的原位超限，请重新 --save")

leader = make_teleoperator_from_config(NeroLeaderConfig(id="nero_leader")); leader.connect()
stop = threading.Event(); follow_th = None
try:
    a = leader.get_action(); m0 = [a[k] for k in keys]
    d_mf = max(abs(deg(x - y)) for x, y in zip(m0, q))
    if MODE == "align" and d_mf < 2.0:
        print(f"主从已对齐（最大差 {d_mf:.1f}°），不用动"); raise SystemExit(0)
    print(f"主从最大差 {d_mf:.1f}°：主臂会以约 34°/s 转到从臂姿态（约 {d_mf/34+0.5:.0f} 秒）")
    if origin:
        d_o = max(abs(deg(x - y)) for x, y in zip(q, origin))
        print(f"之后从臂用 move_j 回原位（最大 {d_o:.1f}°），主臂一路跟随")
    if input("主臂旁有人、两台手臂运动范围内没人？输入 yes 开始：").strip() != "yes": raise SystemExit("已取消")

    leader.set_manual_control(False)
    def follow():
        while not stop.is_set():
            leader.send_feedback(dict(zip(keys, fq()))); time.sleep(1 / 30)
    follow_th = threading.Thread(target=follow, daemon=True); follow_th.start()
    gap = lambda: max(abs(deg(x - y)) for x, y in zip(leader._read_master_joints(), fq()))
    print("1/3 主臂对齐从臂…")
    if not wait_until(lambda: gap() < 1.0, d_mf / 30 + 8, "主臂没能对齐从臂"): raise SystemExit(1)
    print(f"  已对齐（差 {gap():.2f}°）")

    if origin:
        print("2/3 从臂回原位（主臂跟随）…")
        set_gate(True)
        try:
            msg = JointState(); msg.name = names; msg.position = list(origin)
            for _ in range(5): pub.publish(msg); time.sleep(0.05)
            ok = wait_until(lambda: max(abs(deg(x - y)) for x, y in zip(fq(), origin)) < 0.5, 25, "从臂没到原位")
        finally:
            set_gate(False)
        if not ok: raise SystemExit(1)
        wait_until(lambda: gap() < 1.0, 8, "主臂没跟上")
        print(f"  已到原位，主从差 {gap():.2f}°")
    print("3/3 主臂切回示教模式")
finally:
    stop.set()
    if follow_th: follow_th.join(1)
    leader.disconnect()      # 自动切回示教
    ex.shutdown(); th.join(2); node.destroy_node(); rclpy.shutdown()
print("初始化完成：可以开始录制")
