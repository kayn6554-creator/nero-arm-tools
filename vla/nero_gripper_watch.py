#!/usr/bin/env python3
"""只读：看夹爪“有没有收到合拢指令”和“实际合没合上”。不发任何指令。

两种用法（系统 Python，source ROS + agx_arm_ws）：
  /usr/bin/python3 ~/nero_gripper_watch.py            # 实时：开着它，去网页跑一次自动运行，结束后 Ctrl+C
  /usr/bin/python3 ~/nero_gripper_watch.py --last     # 离线：看最近一次自动运行里模型每块输出的夹爪目标

实时模式每 0.5 s 打一行：发给从臂的夹爪指令（/control/joint_states）、从臂实际开度和力（/feedback/gripper_status）。
Ctrl+C 后打印判断，并存 CSV 到 ~/gripper_watch_<时间>.csv。"""
import csv, glob, json, sys, threading, time
from pathlib import Path


def last_run():
    import numpy as np
    runs = sorted(glob.glob("/data/panhe/vla_runs/*/live/chunks.npz"))
    if not runs:
        sys.exit("没有自动运行记录")
    f = runs[-1]
    z = np.load(f, allow_pickle=True)
    st, acts = z["state"], z["actions"]
    print(f"最近一次：{Path(f).parent}")
    print("块  开始时实际开度   模型输出的夹爪目标（这一块 30 步：开头 → 最小 → 结尾）")
    for i in range(len(acts)):
        a = np.asarray(acts[i], float)[:, 7] * 1000
        s = float(st[i][7]) * 1000
        flag = "  ← 模型要合拢" if a.min() < s - 10 else ""
        print(f"{i:2d}   {s:6.1f} mm       {a[0]:6.1f} → {a.min():6.1f} → {a[-1]:6.1f} mm{flag}")
    print("说明：模型输出 < 实际开度很多 = 模型确实在让夹爪合拢；如果从来没有，说明模型没发合拢指令。")


def live():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from rosidl_runtime_py.utilities import get_message

    rows, cur = [], {"cmd": None, "cmd_t": 0.0, "effort": None, "width": None, "force": None, "fb_pos": None}
    t0 = time.time()
    rclpy.init()
    n = Node("nero_gripper_watch")

    def on_cmd(m):
        if "gripper" in m.name:
            i = list(m.name).index("gripper")
            cur["cmd"] = float(m.position[i]); cur["cmd_t"] = time.time()
            cur["effort"] = float(m.effort[i]) if len(m.effort) > i else None

    def on_fb(m):
        if "gripper" in m.name:
            cur["fb_pos"] = float(m.position[list(m.name).index("gripper")])

    def on_gs(m):
        cur["width"], cur["force"] = float(m.width), float(m.force)
        rows.append([round(time.time() - t0, 3), cur["cmd"], cur["effort"], cur["width"], cur["force"], cur["fb_pos"],
                     round(time.time() - cur["cmd_t"], 2) if cur["cmd_t"] else None])

    n.create_subscription(JointState, "/control/joint_states", on_cmd, 10)
    n.create_subscription(JointState, "/feedback/joint_states", on_fb, qos_profile_sensor_data)
    deadline = time.time() + 5   # 夹爪状态的消息类型从话题上查，不写死
    while time.time() < deadline:
        types = dict(n.get_topic_names_and_types())
        if "/feedback/gripper_status" in types: break
        rclpy.spin_once(n, timeout_sec=0.2)
    else:
        sys.exit("找不到 /feedback/gripper_status（驱动没带夹爪？）")
    n.create_subscription(get_message(types["/feedback/gripper_status"][0]), "/feedback/gripper_status", on_gs, qos_profile_sensor_data)
    threading.Thread(target=rclpy.spin, args=(n,), daemon=True).start()
    print("看夹爪中（只读）……去网页跑一次，结束后回这里按 Ctrl+C")
    print("  时间   指令开度  指令力   实际开度   实际力   （指令多久前收到）")
    try:
        while True:
            time.sleep(0.5)
            c, w, f = cur["cmd"], cur["width"], cur["force"]
            age = time.time() - cur["cmd_t"] if cur["cmd_t"] else None
            e = cur["effort"]
            print(f"{time.time() - t0:6.1f}s  " + (f"{c * 1000:6.1f} mm" if c is not None else "   没有  ") + (f"  {e:4.1f} N" if e is not None else "      ")
                  + (f"  {w * 1000:6.1f} mm" if w is not None else "    没有 ") + (f"  {f:+.2f}" if f is not None else "")
                  + (f"   ({age:.1f}s 前)" if age is not None and age > 1 else ""), flush=True)
    except KeyboardInterrupt:
        pass
    fn = Path.home() / time.strftime("gripper_watch_%m%d_%H%M%S.csv")
    with open(fn, "w", newline="") as fp:
        w = csv.writer(fp); w.writerow(["t", "cmd_m", "cmd_effort", "width_m", "force", "feedback_pos", "cmd_age_s"]); w.writerows(rows)
    print(f"\n存了 {len(rows)} 行 → {fn}")
    cmds = [r for r in rows if r[1] is not None]
    if not cmds:
        print("判断：整段时间都没收到夹爪指令 → 不是夹爪卡住，是根本没发（录制 / 模型程序没在控制，或者走的不是 /control/joint_states）")
        return
    lo = min(r[1] for r in cmds)
    closing = [r for r in cmds if r[3] is not None and r[1] < r[3] - 0.010]
    eff = sorted({r[2] for r in cmds if r[2] is not None})
    print(f"指令开度最小 {lo * 1000:.1f} mm；实际开度最小 {min(r[3] for r in rows if r[3] is not None) * 1000:.1f} mm；指令里的夹持力 {eff} N")
    if lo > 0.08:
        print("判断：指令一直是张开的（> 80 mm）→ 模型 / 程序没让夹爪合拢")
    elif closing:
        stuck = [r for r in closing if r[2] and abs(r[4] or 0) >= 0.9 * r[2]]
        print(f"判断：有 {len(closing) / 30:.1f} s 左右“指令比实际更合拢 10 mm 以上”"
              + ("，而且实际力已经顶到指令给的夹持力 → 夹爪在用力但合不上：夹住了物体，或者夹持力太小推不动（看指令力是不是只有 1 N）" if stuck
                 else "，但力不大 → 指令发了、夹爪没跟上，看驱动日志"))
    else:
        print("判断：指令要合拢，实际也跟着合拢了 → 夹爪正常")


if __name__ == "__main__":
    last_run() if "--last" in sys.argv else live()
