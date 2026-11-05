#!/usr/bin/env python3
"""手动执行模型输出：从 nero_vla_client 的 dry 记录里取一块动作，先打印，确认后慢速执行前 N 步。

用法：
  python ~/nero_replay_chunk.py                       # 最近一次 dry 记录里“变化最大”的一块，只打印
  python ~/nero_replay_chunk.py --chunk 1 --steps 10 --slow 2 --go
      --chunk  第几块（默认选块内变化最大的）
      --steps  只执行前几步（10 Hz，10 步 = 1 s 的动作）
      --slow   放慢倍数（2 = 用 2 倍时间走完）
      --go     真正执行（会要求输入 yes）；不加只打印
安全：动作块是按“记录时的姿态”算的，执行前检查从臂当前位置与记录时相差 ≤ 2°，否则拒绝；
下发经 lerobot_robot_nero 插件（限位、单步限幅、对齐 5°、跟踪误差 15°）；Ctrl+C / 结束后关锁 + 原地保持。
"""
import argparse
import glob
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nero_vla_client import GRIPPER_MAX_M, JOINT_KEYS, obs_to_state  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", default=None, help="chunks.npz；默认最近一次 dry 记录")
    p.add_argument("--chunk", type=int, default=None)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--slow", type=float, default=2.0)
    p.add_argument("--action-hz", type=float, default=10.0)
    p.add_argument("--control-hz", type=float, default=30.0)
    p.add_argument("--max-start-dev-deg", type=float, default=2.0)
    p.add_argument("--go", action="store_true")
    args = p.parse_args()

    f = args.file or sorted(glob.glob("/data/panhe/vla_runs/*/dry/chunks.npz"))[-1]
    d = np.load(f, allow_pickle=True)
    states, actions = d["state"], d["actions"]
    spans = [np.abs(np.degrees(a[:, :7] - a[0, :7])).max() for a in actions]
    k = args.chunk if args.chunk is not None else int(np.argmax(spans))
    s0, a = states[k].astype(float), np.asarray(actions[k], float)
    n = min(args.steps, len(a))

    np.set_printoptions(precision=1, suppress=True, linewidth=140)
    print(f"文件 {f}\n第 {k} 块（共 {len(actions)} 块，各块最大变化 {np.round(spans, 1).tolist()}°）")
    print("记录时姿态(°):", np.degrees(s0[:7]), f" 夹爪 {s0[7]*1000:.1f} mm")
    print("步   j1..j7 相对记录时姿态(°)                                   夹爪mm   ← 只执行到 *")
    for i in range(len(a)):
        mark = "*" if i == n - 1 else " "
        print(f"{i:2d}{mark}", np.degrees(a[i, :7] - s0[:7]), f"{a[i, 7]*1000:6.1f}")
    dur = n / args.action_hz * args.slow
    print(f"\n将执行前 {n} 步，用时约 {dur:.1f} s（放慢 {args.slow} 倍）；"
          f"第 {n-1} 步相对现在最大 {np.abs(np.degrees(a[n-1, :7] - s0[:7])).max():.1f}°")
    if not args.go:
        print("（没加 --go，只打印）")
        return 0

    from lerobot_robot_nero import NeroFollower, NeroFollowerConfig
    robot = NeroFollower(NeroFollowerConfig(id="nero", read_only=False, cameras={}))
    robot.connect()   # 非只读：插件打开控制锁
    try:
        cur = obs_to_state(robot.get_observation()).astype(float)
        dev = np.abs(np.degrees(cur[:7] - s0[:7])).max()
        print(f"从臂当前与记录时姿态最大相差 {dev:.2f}°")
        if dev > args.max_start_dev_deg:
            print(f"✗ 超过 {args.max_start_dev_deg}°：手臂动过了，这块动作不再适用，重新跑 dry")
            return 1
        print("⚠ 从臂将运动。手放实体急停。")
        if input("输入 yes 执行：").strip() != "yes":
            print("已取消")
            return 1
        sub = max(1, round(args.control_hz / args.action_hz * args.slow))
        dt = 1.0 / args.control_hz
        prev, t_next = cur, time.monotonic()
        for i in range(n):
            for j in range(1, sub + 1):
                tgt = prev + (a[i] - prev) * (j / sub)
                act = dict(zip(JOINT_KEYS, tgt[:7].tolist()))
                act["gripper.pos"] = float(np.clip(tgt[7], 0.0, GRIPPER_MAX_M))
                robot.send_action(act)
                t_next += dt
                time.sleep(max(0.0, t_next - time.monotonic()))
            prev = a[i]
        time.sleep(0.5)
        end = obs_to_state(robot.get_observation()).astype(float)
        print("执行完。目标−实际(°):", np.degrees(a[n-1, :7] - end[:7]),
              f" 夹爪 目标 {a[n-1,7]*1000:.1f} / 实际 {end[7]*1000:.1f} mm")
        return 0
    except KeyboardInterrupt:
        print("Ctrl+C：停止")
        return 130
    finally:
        robot.disconnect()   # 关锁 + 原地保持


if __name__ == "__main__":
    sys.exit(main())
