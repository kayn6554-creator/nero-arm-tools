#!/usr/bin/env python3
"""从臂回到主臂姿态：读主臂（can1，示教模式，只读）的 7 个关节角，让从臂平滑走过去，夹爪张到给定开度。
用来把每次模型测试的起始姿态统一成主臂现在摆的样子。

走法：7 个关节同时线性插值（不用驱动的 move_j，那个是一个关节一个关节转，中途会下沉），30 Hz 经 lerobot_robot_nero 插件下发，
插件的限位、单步限幅、对齐检查、跟踪误差检查照常生效。如果末端离台面太低，先在原地竖直抬高再过去。
执行前按整条路径算末端和肘 / 腕离台面的高度（台面平面同 nav_guard），会更低就拒绝。

用法（evo-rl 环境）：
  python ~/nero_goto_leader.py            # 打印计划，输入 yes 才执行
  python ~/nero_goto_leader.py --dry      # 只打印，不动
参数：--speed-deg 最快关节速度（默认 15°/s）、--gripper-mm 夹爪开度（默认 93，-1 = 跟主臂夹爪）、--lift-cm 先抬高多少（默认 8）"""
import argparse, math, sys, time
from pathlib import Path
import numpy as np

IK_DIR = "/data/panhe/ik_click"
JOINT_KEYS = [f"joint{i}.pos" for i in range(1, 8)]


class Kin:
    """只用 numpy 的正 / 逆运动学（evo-rl 环境没有 scipy）。MDH 参数来自 pyAgxArm，TCP 偏移来自 arm_config.json，
    结果和学长 downward_ik.DownwardIK.fk 一致（11-03 本地比对过，误差 0）。"""
    def __init__(self, tcp_offset):
        from pyAgxArm.utiles.mdh_kinematics import get_mdh
        self.mdh = get_mdh("nero")
        self.offset = np.asarray(tcp_offset[:3], float)

    def _T(self, q):
        T = np.eye(4)
        for (d, a_, al, off), qi in zip(self.mdh, q):
            th = qi + off; ca, sa, ct, st = math.cos(al), math.sin(al), math.cos(th), math.sin(th)
            T = T @ np.array([[ct, -st, 0, a_], [ca * st, ca * ct, -sa, -sa * d], [sa * st, sa * ct, ca, ca * d], [0, 0, 0, 1]])
        return T

    def fk(self, q):
        T = self._T(q)
        return T[:3, 3] + T[:3, :3] @ self.offset, T[:3, :3]

    def lift(self, q0, dz, lim, iters=150):
        """末端竖直抬高 dz、夹爪指向不变：阻尼最小二乘，数值雅可比，避开限位。"""
        p0, R0 = self.fk(q0); pt = p0 + np.array([0, 0, dz]); q = np.array(q0, float)
        def err(q):
            p, R = self.fk(q)
            return np.r_[pt - p, np.cross(R[:, 0], R0[:, 0])]   # 只保持夹爪指向（工具 x 轴），绕指向的转角不管
        for _ in range(iters):
            e = err(q)
            if np.linalg.norm(e[:3]) < 0.002 and np.linalg.norm(e[3:]) < math.radians(2): return q
            J = np.zeros((6, 7))
            for j in range(7):
                dq = np.zeros(7); dq[j] = 1e-5
                J[:, j] = -(err(q + dq) - e) / 1e-5
            at_lo, at_hi = q <= lim[:, 0] + 1e-6, q >= lim[:, 1] - 1e-6
            Jf = J.copy()
            dq = Jf.T @ np.linalg.solve(Jf @ Jf.T + 1e-3 * np.eye(6), e)
            blocked = (at_lo & (dq < 0)) | (at_hi & (dq > 0))   # 已在限位、还要往外走的关节冻结后重解（7 轴有冗余）
            if blocked.any():
                Jf[:, blocked] = 0
                dq = Jf.T @ np.linalg.solve(Jf @ Jf.T + 1e-3 * np.eye(6), e)
            dq *= min(1.0, 0.05 / max(1e-9, np.abs(dq).max()))   # 每步最多 0.05 rad，线性化才可靠
            q = np.clip(q + dq, lim[:, 0], lim[:, 1])
        raise RuntimeError("抬高的逆解没收敛")


def read_leader():
    from lerobot_teleoperator_nero import NeroLeader, NeroLeaderConfig
    ld = NeroLeader(NeroLeaderConfig(id="nero_leader", positioning_follow=False))
    ld.connect()
    try:
        a = ld.get_action()
    finally:
        ld.disconnect()
    return np.array([a[k] for k in JOINT_KEYS], float), a.get("gripper.pos")


def path_clearance(ik, ng, qs):
    """沿一串关节角，算末端（TCP / 法兰）和肘、腕离台面的最低高度（m）。"""
    tcp_min = link_min = 9.0
    for q in qs:
        t, l = ng._clear(ik, q)
        tcp_min, link_min = min(tcp_min, t), min(link_min, l)
    return tcp_min, link_min


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--speed-deg", type=float, default=15.0)
    p.add_argument("--gripper-mm", type=float, default=93.0)
    p.add_argument("--lift-cm", type=float, default=8.0)
    p.add_argument("--low-cm", type=float, default=12.0, help="末端离台面低于这个高度时先抬高")
    p.add_argument("--min-link-cm", type=float, default=3.0)
    p.add_argument("--dry", action="store_true")
    a = p.parse_args()
    if not 3 <= a.speed_deg <= 30: sys.exit("--speed-deg 应在 3~30")

    sys.path.insert(0, IK_DIR)
    import json
    import nav_guard as ng
    ik = Kin(json.loads(Path(IK_DIR, "arm_config.json").read_text())["tcp_offset"])

    q_lead, g_lead = read_leader()
    print("主臂姿态 (°):", np.round(np.degrees(q_lead), 1).tolist(), f" 主臂夹爪 {(g_lead or 0) * 1000:.0f} mm")

    from lerobot_robot_nero import NeroFollower, NeroFollowerConfig
    robot = NeroFollower(NeroFollowerConfig(id="nero", read_only=a.dry, cameras={}))
    robot.connect()
    try:
        obs = robot.get_observation()
        q0 = np.array([obs[k] for k in JOINT_KEYS], float); g0 = float(obs.get("gripper.pos", 0.0))
        print("从臂现在 (°):", np.round(np.degrees(q0), 1).tolist(), f" 夹爪 {g0 * 1000:.0f} mm")
        lim = np.array(robot.config.joint_limits)
        if np.any(q_lead < lim[:, 0]) or np.any(q_lead > lim[:, 1]):
            sys.exit("主臂姿态超出从臂限位，先把主臂摆进限位内")
        g_t = (g_lead if a.gripper_mm < 0 else a.gripper_mm / 1000) or 0.0
        g_t = min(max(g_t, 0.0), 0.1)

        # 路线：现在 →（太低时先竖直抬高）→ 主臂姿态
        way = [q0]
        clear0, _ = path_clearance(ik, ng, [q0])
        if clear0 < a.low_cm / 100:
            try:
                lift = ik.lift(q0, a.lift_cm / 100, lim)
                way.append(lift); print(f"末端离台面 {clear0 * 100:.1f} cm，先竖直抬高 {a.lift_cm:.0f} cm")
            except RuntimeError as e:
                print(f"抬高的 IK 无解（{e}），直接过去")
        way.append(q_lead)
        segs, qs = [], []
        for s, t in zip(way[:-1], way[1:]):
            n = max(2, int(math.degrees(np.abs(t - s).max()) / a.speed_deg * 30))
            seg = [s + (t - s) * k / n for k in range(1, n + 1)]
            segs.append(seg); qs += seg
        tcp_min, link_min = path_clearance(ik, ng, [q0] + qs)
        need = min(clear0, 0.05) - 0.01
        dur = len(qs) / 30
        print(f"路线 {len(way) - 1} 段，用时约 {dur:.1f} s，最大关节变化 {math.degrees(np.abs(q_lead - q0).max()):.0f}°；"
              f"途中末端离台面最低 {tcp_min * 100:.1f} cm，肘 / 腕最低 {link_min * 100:.1f} cm；夹爪 {g0 * 1000:.0f} → {g_t * 1000:.0f} mm")
        if tcp_min < need:
            sys.exit(f"拒绝：途中末端会低到 {tcp_min * 100:.1f} cm（不能低于 {need * 100:.1f} cm）。先用遥操作把从臂抬高一些")
        if link_min < a.min_link_cm / 100:
            sys.exit(f"拒绝：途中肘 / 腕会低到 {link_min * 100:.1f} cm")
        if a.dry:
            print("（--dry：不执行）"); return 0
        if input("从臂将走到主臂姿态。手放实体急停，输入 yes 执行：").strip() != "yes":
            print("已取消"); return 1
        k, total, t_next = 0, len(qs), time.monotonic()
        for q in qs:
            k += 1
            act = dict(zip(JOINT_KEYS, q.tolist()))
            act["gripper.pos"] = float(g0 + (g_t - g0) * k / total)
            robot.send_action(act)
            t_next += 1 / 30
            time.sleep(max(0.0, t_next - time.monotonic()))
        time.sleep(0.5)
        obs = robot.get_observation()
        q1 = np.array([obs[k] for k in JOINT_KEYS], float)
        print(f"完成：和主臂最大差 {math.degrees(np.abs(q1 - q_lead).max()):.1f}°，夹爪 {obs.get('gripper.pos', 0) * 1000:.0f} mm")
        return 0
    except KeyboardInterrupt:
        print("Ctrl+C：停止"); return 130
    finally:
        robot.disconnect()   # 关锁 + 原地保持


if __name__ == "__main__":
    sys.exit(main())
