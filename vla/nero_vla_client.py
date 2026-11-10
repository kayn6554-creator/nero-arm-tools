#!/usr/bin/env python3
"""NERO 单臂 VLA 推理客户端：读从臂 + 两路相机 → 远程模型（openpi websocket）→ 动作块 → 从臂插件执行。

三种模式（默认最安全的 check）：
  check  不连模型、不动臂：检查从臂状态、两相机配对，打印一次请求的形状
  dry    连模型推理、不动臂：打印动作块、首帧偏差、推理耗时（从臂插件只读）
  live   连模型并执行动作：需要手放实体急停，启动时输入 yes

请求格式（发给模型服务，和 LeRobot 数据集一致）：
  state   float32[8]       joint1..joint7 (rad) + gripper (m, 0~0.1)，即 observation.state
  images  {front, wrist}   uint8，默认 CHW 3x480x640（--image-layout hwc 可改），RGB
  prompt  str              任务描述，默认取 ~/nero_tasks.json 里的 task（与录制时的 single_task 相同）
返回：
  actions float[T, 8]      绝对关节目标 (rad) + 夹爪 (m)，T 步，每步 1/--action-hz 秒（数据集 10 Hz）

周末实验（2026-11）新增：
  --step-file F      每次推理前读 F 里的当前步骤（1~4，测试台按“本步完成”时更新），记进记录
  --step-prompt      把步骤文本接到指令后面（M1/M2 这类带步骤训练的模型）；格式见 nero_steps.py
  --variant NAME     实验变体标签（E0 / E1 / M1 ...），记进 summary
  每次推理另存头部/腕部图像（images/），执行时按 --feedback-every 记录从臂实际位置（feedback.npz），
  每块写 live_status.json 给测试台显示；夹爪越界不再拒绝整块，裁到 0~0.1 并计数

安全：
  - 动作经 lerobot_robot_nero 插件下发：限位、单步限幅、对齐检查(5°)、跟踪误差(15°)、状态过期都由插件把关
  - 本脚本另外检查：metadata 维度、动作有限、相邻两步跳变、首帧离当前位置的距离
  - 网页“定位开”时插件不执行，本脚本暂停等待
  - Ctrl+C / 任何异常：插件 disconnect → 关控制锁 + 原地保持
"""
from __future__ import annotations

import argparse
import functools
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nero_steps import CLIENT_TASKS, step_prompt  # noqa: E402  与训练代码同一份定义

log = logging.getLogger("nero_vla")

JOINT_KEYS = [f"joint{i}.pos" for i in range(1, 8)]
STATE_DIM = 8
GRIPPER_MAX_M = 0.100


# ---------------------------------------------------------------- 协议（与 openpi_client 相同）
def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


class PolicyClient:
    """openpi-websocket-v1：连上后服务端先发 metadata；之后每次 send 一个请求、recv 一个结果。"""

    def __init__(self, host: str, port: int, timeout_s: float = 10.0):
        import msgpack
        import websockets.sync.client as wsc

        for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(k, None)
        self._packb = functools.partial(msgpack.packb, default=_pack_array)
        self._unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)
        self.uri = f"ws://{host}:{port}"
        self._ws = wsc.connect(self.uri, compression=None, max_size=None, open_timeout=timeout_s,
                               ping_interval=20, ping_timeout=60)
        self.metadata = self._unpackb(self._ws.recv())

    def infer(self, request: dict, timeout_s: float | None = None) -> dict:
        self._ws.send(self._packb(request))
        try:
            resp = self._ws.recv(timeout=timeout_s)
        except TypeError:          # 旧版 websockets 的 recv 没有 timeout 参数
            resp = self._ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"模型服务报错：\n{resp}")
        return self._unpackb(resp)

    def close(self):
        try:
            self._ws.close()
        except Exception:
            pass


# ---------------------------------------------------------------- 请求 / 动作
def to_layout(img: np.ndarray, layout: str) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"图像应为 HWC RGB，实际 {img.shape}")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img.transpose(2, 0, 1) if layout == "chw" else img)


def obs_to_state(obs: dict) -> np.ndarray:
    s = np.array([obs[k] for k in JOINT_KEYS] + [obs.get("gripper.pos", 0.0)], dtype=np.float32)
    if s.shape != (STATE_DIM,) or not np.all(np.isfinite(s)):
        raise ValueError(f"状态异常：{s}")
    return s


def build_request(state, head, wrist, prompt, args) -> dict:
    return {
        "state": state.astype(np.float32),
        "images": {args.head_key: to_layout(head, args.image_layout),
                   args.wrist_key: to_layout(wrist, args.image_layout)},
        "prompt": prompt,
    }


def build_piper_compat_request(state, head, wrist, prompt, args) -> dict:
    """仅 dry 模式协议测试用：把 NERO 观测塞进当前双臂 Piper 服务的格式。输出没有物理意义。"""
    s = np.zeros(14, np.float32)
    s[:6] = state[:6]
    s[6] = np.clip(state[7] / GRIPPER_MAX_M, 0, 1)
    return {
        "state": s,
        "images": {"cam_high": to_layout(head, "chw"), "cam_left_wrist": to_layout(wrist, "chw"),
                   "cam_right_wrist": np.zeros_like(to_layout(wrist, "chw"))},
        "prompt": prompt,
    }


def parse_actions(resp: dict, dim: int) -> np.ndarray:
    a = np.asarray(resp["actions"], dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[1] != dim or not np.all(np.isfinite(a)):
        raise ValueError(f"动作应为有限的 [T, {dim}]，实际 {a.shape}")
    return a


def check_chunk(actions: np.ndarray, state: np.ndarray, args) -> list[str]:
    """返回问题列表；空列表 = 可以执行。"""
    problems = []
    first = np.abs(actions[0, :7] - state[:7]).max()
    if first > math.radians(args.max_first_dev_deg):
        problems.append(f"首帧离当前位置 {math.degrees(first):.1f}° > {args.max_first_dev_deg}°")
    if len(actions) > 1:
        jump = np.abs(np.diff(actions[:, :7], axis=0)).max()
        if jump > math.radians(args.max_jump_deg):
            problems.append(f"相邻两步最大跳变 {math.degrees(jump):.1f}° > {args.max_jump_deg}°")
    g = actions[:, 7]
    # 11-03：夹爪略超 0~0.1 很常见（中位数约 -2 mm），下发时裁剪并计数；只有离谱的（单位错）才拒绝
    if g.min() < -0.05 or g.max() > GRIPPER_MAX_M + 0.05:
        problems.append(f"夹爪远超 0~{GRIPPER_MAX_M} m：[{g.min():.3f}, {g.max():.3f}]（单位对吗？）")
    return problems


def describe_chunk(actions: np.ndarray, state: np.ndarray) -> str:
    d0 = np.degrees(actions[0, :7] - state[:7])
    span = np.degrees(actions[:, :7].max(0) - actions[:, :7].min(0))
    return (f"T={len(actions)}  首帧−当前(°)={np.round(d0, 1).tolist()}  "
            f"块内变化幅度(°)={np.round(span, 1).tolist()}  夹爪 {actions[0, 7]*1000:.0f}→{actions[-1, 7]*1000:.0f} mm")


# ---------------------------------------------------------------- 硬件
class _Frame:
    """缓存里的一帧：回调只存原始消息，用到时才解码（回调越短，越不容易排队）。"""
    __slots__ = ("msg", "stamp_ms", "recv", "_rgb")

    def __init__(self, msg, recv):
        self.msg, self.recv, self._rgb = msg, recv, None
        self.stamp_ms = msg.header.stamp.sec * 1000.0 + msg.header.stamp.nanosec / 1e6

    @property
    def color(self) -> np.ndarray:
        if self._rgb is None:
            m = self.msg
            enc = m.encoding.lower()
            if enc not in ("rgb8", "bgr8"):
                raise ValueError(f"不支持的编码 {m.encoding}")
            img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, : m.width * 3].reshape(m.height, m.width, 3)
            self._rgb = np.array(img[:, :, ::-1] if enc == "bgr8" else img)
        return self._rgb


class LatestCam:
    """轻量 ROS 图像订阅：可靠模式、队列深度 1（处理不过来就丢旧帧，不排队），只缓存最近几帧的原始消息。
    插件的 RosImageCamera 队列深度 10，在本进程里会积压到约 0.5 s（11-02 实测）。"""

    def __init__(self, topic: str, width: int = 640, height: int = 480, buffer_len: int = 8):
        from collections import deque
        import threading
        self.topic, self.width, self.height = topic, width, height
        self._buf = deque(maxlen=buffer_len)
        self._lock = threading.Lock()
        self._node = self._executor = self._thread = None

    def __str__(self):
        return f"LatestCam({self.topic})"

    def connect(self, timeout_s: float = 5.0):
        import threading
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from rclpy.signals import SignalHandlerOptions
        from sensor_msgs.msg import Image

        if not rclpy.ok():
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self._node = Node("nero_vla_cam_" + "".join(c if c.isalnum() else "_" for c in self.topic).strip("_"))
        qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE)
        self._node.create_subscription(Image, self.topic, self._on_image, qos)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + timeout_s
        while not self._buf:
            if time.monotonic() > deadline:
                self.disconnect()
                raise ConnectionError(f"{timeout_s}s 内没收到 {self.topic}：相机驱动在运行吗？")
            time.sleep(0.02)
        fr = self._buf[-1]
        if (fr.msg.width, fr.msg.height) != (self.width, self.height):
            raise ValueError(f"{self.topic} 画面 {fr.msg.width}x{fr.msg.height} ≠ {self.width}x{self.height}")
        log.info("%s connected.", self)

    def _on_image(self, msg):
        fr = _Frame(msg, time.monotonic())
        with self._lock:
            self._buf.append(fr)

    def recent(self, n: int = 3) -> list:
        with self._lock:
            frames = list(self._buf)[-n:][::-1]
        if not frames or time.monotonic() - frames[0].recv > 0.5:
            raise TimeoutError(f"{self} 超过 0.5 s 没有新画面")
        return frames

    def read_nearest(self, stamp_ms: float, min_recv: float = 0.0):
        with self._lock:
            frames = list(self._buf)
        if min_recv > 0:
            frames = [f for f in frames if f.recv >= min_recv] or frames[-1:]
        if not frames:
            raise TimeoutError(f"{self} 没有画面")
        return min(frames, key=lambda f: abs(f.stamp_ms - stamp_ms))

    def disconnect(self):
        if self._executor is not None:
            self._executor.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._node is not None:
            self._node.destroy_node()
        self._node = self._executor = self._thread = None
        log.info("%s disconnected.", self)


class NeroIO:
    """从臂（lerobot_robot_nero）+ 两个 ROS 相机（lerobot_camera_ros），与录制时同样的话题和配对方式。"""

    def __init__(self, args, read_only: bool):
        from lerobot_robot_nero import NeroFollower, NeroFollowerConfig

        self.args = args
        self.robot = NeroFollower(NeroFollowerConfig(id="nero", read_only=read_only, cameras={}))
        if args.cam_impl == "plugin":   # 录制插件的相机类（队列 10，可带深度），留作对比
            from lerobot_camera_ros import RosImageCamera, RosImageCameraConfig
            self.head = RosImageCamera(RosImageCameraConfig(
                topic=args.head_topic, depth_topic=args.head_depth_topic or None, width=640, height=480, fps=15))
            self.wrist = RosImageCamera(RosImageCameraConfig(
                topic=args.wrist_topic, depth_topic=args.wrist_depth_topic or None, width=640, height=480, fps=30))
        else:
            self.head, self.wrist = LatestCam(args.head_topic), LatestCam(args.wrist_topic)
        self.connected = []

    def connect(self):
        self.robot.connect()
        self.connected.append(self.robot)
        for cam in (self.head, self.wrist):
            cam.connect()
            self.connected.append(cam)
        time.sleep(1.0)   # 两路订阅不是同时建立，等缓存都有数据再配对（问题 #40）

    def disconnect(self):
        for dev in reversed(self.connected):   # 先关相机，最后从臂（关锁 + 原地保持）
            try:
                dev.disconnect()
            except Exception as e:
                log.error("断开 %s 出错：%s", dev, e)
        self.connected.clear()

    @property
    def positioning(self) -> bool:
        return bool(getattr(self.robot, "_positioning", False))

    def observe(self, max_sync_ms: float = 20.0, retry_s: float = 1.0, after: float = 0.0):
        """配对超过 max_sync_ms 时在 retry_s 内重试，返回最好的一组。after：只用这个时刻（monotonic）之后收到的帧。"""
        deadline = time.monotonic() + retry_s
        best = None
        while True:
            got = self._observe_once(after)
            if best is None or got[3] < best[3]:
                best = got
            if best[3] <= max_sync_ms or time.monotonic() > deadline:
                return best
            time.sleep(0.03)

    def _observe_once(self, after: float = 0.0):
        """返回 (state[8], head RGB, wrist RGB, 两相机时间差 ms, 头部画面年龄 ms, 腕部画面年龄 ms)。
        after > 0 时两路都只在“after 之后收到”的帧里挑（11-06：以前腕部按时间戳就近配对，可能挑到停稳前的旧帧）。"""
        best = None
        heads = self.head.recent(3)
        if after > 0 and hasattr(heads[0], "recv"):
            heads = [f for f in heads if f.recv >= after] or heads[:1]
        for h in heads:
            w = (self.wrist.read_nearest(h.stamp_ms, min_recv=after) if isinstance(self.wrist, LatestCam)
                 else self.wrist.read_nearest(h.stamp_ms))
            diff = abs(w.stamp_ms - h.stamp_ms)
            if best is None or diff < best[0]:
                best = (diff, h, w)
            if diff <= 18.0:
                best = (diff, h, w)
                break
        diff, h, w = best
        obs = self.robot.get_observation()
        # 画面年龄按“本进程收到它到现在”算；相机的 ROS 时间戳会漂（11-02 实测腕部比系统时钟快），不拿来判断新旧
        now = time.monotonic()
        return obs_to_state(obs), h.color, w.color, diff, (now - h.recv) * 1000, (now - w.recv) * 1000


# ---------------------------------------------------------------- 执行
def execute_chunk(io: NeroIO, actions: np.ndarray, start: np.ndarray, args, fb: list | None = None,
                  chunk: int = -1, stats: dict | None = None) -> str:
    """按 --control-hz 插值执行前 --exec-steps 步；返回结束原因。
    每一步按“放慢倍数”分配时间；如果某一步关节变化太大、按这个时间走会超过 --max-joint-vel-deg，
    就自动把这一步拉长（多插几帧），保证实际关节速度不超过上限。"""
    base = max(1, round(args.control_hz / args.action_hz * args.slow))
    dt = 1.0 / args.control_hz
    vmax = math.radians(args.max_joint_vel_deg)
    prev = start.astype(np.float64)
    t_next = time.monotonic()
    stretched = []
    for k in range(min(args.exec_steps, len(actions))):
        goal = actions[k]
        need = math.ceil(np.abs(goal[:7] - prev[:7]).max() / (vmax * dt))
        sub = max(base, need)
        if sub > base:
            stretched.append((k, round(sub / base, 1)))
        for j in range(1, sub + 1):
            if io.positioning:
                return "网页定位开启，暂停执行"
            target = prev + (goal - prev) * (j / sub)
            act = dict(zip(JOINT_KEYS, target[:7].tolist()))
            act["gripper.pos"] = float(np.clip(target[7], 0.0, GRIPPER_MAX_M))
            if stats is not None and act["gripper.pos"] != float(target[7]):
                stats["grip_clip"] = stats.get("grip_clip", 0) + 1
            io.robot.send_action(act)
            tick = stats.setdefault("tick", 0) if stats is not None else 0
            if stats is not None:
                stats["tick"] = tick + 1
            if fb is not None and args.feedback_every > 0 and tick % args.feedback_every == 0:
                try:   # 从臂实际位置：区分“模型没预测到位”和“从臂没跟上”（H8）
                    meas = obs_to_state(io.robot.get_observation())
                    fb.append((time.time(), chunk, k, *target[:7], act["gripper.pos"], *meas))
                except Exception as e:
                    stats["fb_err"] = str(e)
            t_next += dt
            time.sleep(max(0.0, t_next - time.monotonic()))
        prev = goal
    if stretched:
        log.info("有 %d 步关节变化较大，自动放慢执行：%s", len(stretched),
                 "，".join(f"第 {k} 步 ×{r}" for k, r in stretched[:6]) + ("…" if len(stretched) > 6 else ""))
    return "ok"


def run(args):
    mode = args.mode
    prompt = args.prompt or load_task_prompt(args.task)
    task_key = CLIENT_TASKS.get(args.task)
    if args.step_prompt and (task_key is None or not args.step_file):
        raise SystemExit("--step-prompt 需要 --step-file，且任务是 weigh / task2 / task3")
    log.info("模式 %s   任务：%s   变体：%s   步骤文本：%s", mode, prompt, args.variant or "-", "开" if args.step_prompt else "关")

    if mode == "live":
        print("\n⚠ live 模式会让从臂按模型输出运动。确认：手放实体急停、录制程序没在运行、网页没在 Move。")
        if input("输入 yes 继续：").strip() != "yes":
            print("已取消")
            return 1

    policy = None
    if mode != "check":
        policy = PolicyClient(args.server_host, args.server_port)
        md = policy.metadata
        log.info("模型服务 %s  metadata=%s", policy.uri, md)
        dim = int(md.get("action_dim", -1))
        if args.piper_compat:
            if mode != "dry":
                raise SystemExit("--piper-compat 只允许在 dry 模式下用（输出没有物理意义）")
            if dim != 14:
                raise SystemExit(f"--piper-compat 需要 14 维服务，当前 {dim}")
        elif dim != STATE_DIM:
            policy.close()
            raise SystemExit(f"模型服务 action_dim={dim}，NERO 需要 {STATE_DIM}。"
                             "服务端还不是 NERO 模型；只想测协议可以用 --mode dry --piper-compat")
        if md.get("action_semantics", "absolute_joint") != "absolute_joint" and not args.piper_compat:
            raise SystemExit(f"只支持绝对关节动作，服务报告 {md.get('action_semantics')}")

    run_dir = Path(args.log_dir) / time.strftime("%Y%m%d_%H%M%S") / mode
    run_dir.mkdir(parents=True, exist_ok=True)
    io = NeroIO(args, read_only=(mode != "live"))
    records, feedback, stats = [], [], {}
    executed, end_reason, t_start, timeouts = 0, "", time.time(), 0
    if args.save_images:
        (run_dir / "images").mkdir(exist_ok=True)
    if mode == "live":
        print(f"RUN_DIR={run_dir}", flush=True)
    try:
        io.connect()
        log.info("从臂和两相机已连接（%s）", "可控制" if mode == "live" else "只读")
        t_end = time.monotonic() + args.max_duration_s
        t_stop = time.monotonic()          # 开始前手臂是静止的
        for chunk in range(args.max_chunks):
            if time.monotonic() > t_end:
                end_reason = f"到达最长时长 {args.max_duration_s:.0f} s"
                log.info(end_reason)
                break
            if io.positioning:
                log.info("网页定位开启中，等待定位关……")
                while io.positioning:
                    time.sleep(0.2)
            if mode == "check":
                check_report(io, args, prompt, run_dir)
                break
            # 机械臂停稳后再取观测：推理期间手臂本来就静止，等 settle_s 让画面追上实际姿态，
            # 这样相机传输延迟（实测 0.1~0.5 s）和两相机时间差都不影响输入
            time.sleep(args.settle_s)
            if not wait_fresh(io, t_stop + args.fresh_after_stop_s):
                end_reason = "手臂停下后 2 s 内没收到新画面，停止"
                log.error(end_reason)
                break
            state, head, wrist, sync_ms, age_h, age_w = io.observe(after=t_stop + args.fresh_after_stop_s)
            step = read_step(args.step_file)
            if args.step_prompt and step == 0:
                log.warning("读不到当前步骤（%s），按第 1 步发送", args.step_file)
            prompt_k = step_prompt(prompt, task_key, max(step, 1)) if args.step_prompt else prompt
            if max(age_h, age_w) > args.max_image_age_ms:
                log.error("画面过期：头部 %.0f ms、腕部 %.0f ms 没有新画面，停止", age_h, age_w)
                break
            if sync_ms > 20.0:
                log.warning("两相机时间戳差 %.1f ms（手臂静止时不影响）", sync_ms)
            build = build_piper_compat_request if args.piper_compat else build_request
            req = build(state, head, wrist, prompt_k, args)
            if args.save_images:
                save_images(run_dir / "images", chunk, head, wrist)

            t0 = time.monotonic()
            try:
                resp = policy.infer(req, timeout_s=args.infer_timeout_s)
            except TimeoutError:
                # 超时后这条连接上可能还会晚到一个旧结果，不能接着用：断开重连，用新的观测再试一次
                timeouts += 1
                log.warning("模型 %.0f s 没有返回（第 %d 次），重连后重试", args.infer_timeout_s, timeouts)
                policy.close()
                if timeouts > args.infer_retries:
                    end_reason = f"模型连续 {timeouts} 次超过 {args.infer_timeout_s:.0f} s 没有返回，停止"
                    log.error(end_reason)
                    policy = None
                    break
                try:
                    policy = PolicyClient(args.server_host, args.server_port)
                except Exception as e:
                    end_reason = f"重连模型服务失败：{e}"
                    log.error(end_reason)
                    policy = None
                    break
                continue
            timeouts = 0
            latency = time.monotonic() - t0
            actions = parse_actions(resp, 14 if args.piper_compat else STATE_DIM)
            if args.piper_compat:   # 只看左臂 6 关节 + 左夹爪，补一个 j7=当前值
                a = np.zeros((len(actions), 8))
                a[:, :6], a[:, 6], a[:, 7] = actions[:, :6], state[6], actions[:, 6] * GRIPPER_MAX_M
                actions = a
            problems = check_chunk(actions, state, args)
            log.info("第 %d 块  步骤 %s  推理 %.2f s  %s", chunk, step, latency, describe_chunk(actions, state))
            records.append(dict(chunk=chunk, t=time.time(), latency=latency, sync_ms=sync_ms,
                                state=state, actions=actions, problems=problems, step=step,
                                age_h=age_h, age_w=age_w, prompt_used=resp.get("prompt_used") or prompt_k))
            write_status(run_dir, records[-1], stats, args)
            if mode == "dry" or chunk == 0:
                log.info("  指令：%s", records[-1]["prompt_used"])

            if mode == "dry":
                for p in problems:
                    log.warning("（dry）如果是 live 会拒绝执行：%s", p)
                time.sleep(args.dry_interval_s)
                continue

            if problems:
                end_reason = "拒绝执行：" + "；".join(problems)
                log.error(end_reason)
                break
            log.info("执行前 %d 步（放慢 %.1f 倍，约 %.1f s）", min(args.exec_steps, len(actions)), args.slow,
                     min(args.exec_steps, len(actions)) / args.action_hz * args.slow)
            reason = execute_chunk(io, actions, state, args, fb=feedback, chunk=chunk, stats=stats)
            t_stop = time.monotonic()
            executed += 1
            if reason != "ok":
                log.warning(reason)
        else:
            end_reason = f"到达最多 {args.max_chunks} 块"
        return 0
    except KeyboardInterrupt:
        end_reason = "人工停止（Ctrl+C / 网页停止 / 急停）"
        log.warning("Ctrl+C：停止")
        return 130
    except Exception as e:
        end_reason = f"出错：{e}"
        raise
    finally:
        io.disconnect()
        if policy is not None:
            policy.close()
        if records or mode == "live":
            (run_dir / "summary.json").write_text(json.dumps(
                dict(args=vars(args), prompt=prompt, mode=mode, started=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start)),
                     duration_s=round(time.time() - t_start, 1), executed_chunks=executed, end_reason=end_reason,
                     variant=args.variant, step_prompt=args.step_prompt, gripper_clipped_ticks=stats.get("grip_clip", 0),
                     feedback_error=stats.get("fb_err"),
                     chunks=[dict(chunk=r["chunk"], latency=round(r["latency"], 3), step=r["step"],
                                  sync_ms=round(r["sync_ms"], 1), age_ms=[round(r["age_h"]), round(r["age_w"])],
                                  prompt_used=r["prompt_used"], problems=r["problems"]) for r in records]),
                ensure_ascii=False, indent=1))
        if records:
            np.savez_compressed(
                run_dir / "chunks.npz",
                state=np.stack([r["state"] for r in records]),
                actions=np.stack([r["actions"] for r in records]) if len({r["actions"].shape for r in records}) == 1
                else np.array([r["actions"] for r in records], dtype=object),
                latency=np.array([r["latency"] for r in records]),
                sync_ms=np.array([r["sync_ms"] for r in records]),
                t=np.array([r["t"] for r in records]),
                step=np.array([r["step"] for r in records]),
            )
        if feedback:
            # 每行：time, chunk, 步序号 k, 下发目标 j1..j7 + 夹爪(裁剪后), 实测 j1..j7 + 夹爪
            np.savez_compressed(run_dir / "feedback.npz", rows=np.array(feedback, dtype=np.float64),
                                columns=np.array(["t", "chunk", "k"] + [f"target_j{i}" for i in range(1, 8)] + ["target_grip"]
                                                 + [f"meas_j{i}" for i in range(1, 8)] + ["meas_grip"]))
            log.info("记录已存 %s", run_dir)


def read_step(path) -> int:
    """测试台写的当前步骤（1~4）；没给文件就是 0。读不到时保持 0，不中断运行。"""
    if not path:
        return 0
    try:
        s = int(Path(path).expanduser().read_text().strip() or 0)
        return s if 1 <= s <= 4 else 0
    except Exception:
        return 0


def save_images(folder: Path, chunk: int, head: np.ndarray, wrist: np.ndarray):
    try:
        from PIL import Image
        Image.fromarray(head).save(folder / f"{chunk:03d}_front.jpg", quality=90)
        Image.fromarray(wrist).save(folder / f"{chunk:03d}_wrist.jpg", quality=90)
    except Exception as e:   # 存图失败不影响运行
        log.warning("存图失败：%s", e)


def write_status(run_dir: Path, r: dict, stats: dict, args):
    """给测试台看的最新一块（每块覆盖写）。"""
    a, s = r["actions"], r["state"]
    try:
        (run_dir / "live_status.json").write_text(json.dumps(dict(
            t=r["t"], chunk=r["chunk"], step=r["step"], variant=args.variant, prompt_used=r["prompt_used"],
            latency=round(r["latency"], 3), sync_ms=round(r["sync_ms"], 1), age_ms=[round(r["age_h"]), round(r["age_w"])],
            move_deg=round(float(np.degrees(np.abs(a[min(args.exec_steps, len(a)) - 1, :7] - s[:7]).max())), 2),
            grip_now_mm=round(float(s[7]) * 1000, 1), grip_pred_mm=[round(float(a[:, 7].min()) * 1000, 1), round(float(a[:, 7].max()) * 1000, 1)],
            grip_clipped_ticks=stats.get("grip_clip", 0), problems=r["problems"]), ensure_ascii=False))
    except Exception as e:
        log.warning("写 live_status 失败：%s", e)


def wait_fresh(io: "NeroIO", after: float, timeout_s: float = 2.0) -> bool:
    """等到两个相机都有 after（monotonic）之后才收到的画面。插件相机类没有收到时刻，直接放行。"""
    deadline = time.monotonic() + timeout_s
    while True:
        ok = True
        for cam in (io.head, io.wrist):
            buf = getattr(cam, "_buf", None)
            if buf is not None and (not buf or buf[-1].recv < after):
                ok = False
        if ok:
            return True
        if time.monotonic() > deadline:
            return False
        time.sleep(0.02)


def check_report(io: NeroIO, args, prompt: str, run_dir: Path, seconds: float = 5.0):
    """check 模式：按 10 Hz 连续取 5 s 观测，统计画面年龄和两相机时间差。"""
    rows, sample = [], None
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        state, head, wrist, sync_ms, age_h, age_w = io._observe_once()
        # 诊断：各相机最新一帧在本进程收到时已经晚了多少（收到时刻 − ROS 时间戳），以及缓存里最新一帧的年龄
        off = time.time() - time.monotonic()
        lat = []
        for cam in (io.head, io.wrist):
            fr = cam.recent(1)[0]
            lat += [(fr.recv + off) * 1000 - fr.stamp_ms, time.time() * 1000 - fr.stamp_ms]
        rows.append((sync_ms, age_h, age_w, *lat))
        sample = (state, head, wrist)
        time.sleep(0.1)
    r = np.array(rows)
    state, head, wrist = sample
    req = build_request(state, head, wrist, prompt, args)
    print(f"state = {np.round(state, 4).tolist()}")
    print("images: " + ", ".join(f"{k} {v.shape} {v.dtype}" for k, v in req["images"].items()))
    print(f"prompt = {prompt!r}")
    for i, name in enumerate(["两相机时间差", "头部距上次收到", "腕部距上次收到",
                              "头部·时间戳到收到", "头部·时间戳到现在", "腕部·时间戳到收到", "腕部·时间戳到现在"]):
        c = r[:, i]
        print(f"{name}: 前 1 s 最大 {c[:10].max():.0f} ms | 之后 中位 {np.median(c[10:]):.0f}  "
              f"95% {np.percentile(c[10:], 95):.0f}  最大 {c[10:].max():.0f} ms")
    print(f"共 {len(r)} 次，之后时间差 > 20 ms 的 {int((r[10:, 0] > 20).sum())} 次，"
          f"画面年龄 > {args.max_image_age_ms:.0f} ms 的 {int((r[10:, 1:] > args.max_image_age_ms).any(1).sum())} 次")
    np.savez_compressed(run_dir / "request_sample.npz", state=state, head=head, wrist=wrist, stats=r)
    print(f"样例已存 {run_dir}/request_sample.npz")


def load_task_prompt(task: str) -> str:
    tasks = json.loads(Path("~/nero_tasks.json").expanduser().read_text())
    if task not in tasks:
        raise SystemExit(f"~/nero_tasks.json 里没有任务 {task}，可选：{list(tasks)}")
    return tasks[task]["task"]


def argument_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["check", "dry", "live"], default="check")
    p.add_argument("--task", default="weigh", help="~/nero_tasks.json 里的预设：weigh / task2 / task3")
    p.add_argument("--prompt", default=None, help="直接给任务描述，覆盖 --task")
    p.add_argument("--server-host", default="172.16.11.115")
    p.add_argument("--server-port", type=int, default=6699, help="6699 = NERO 模型；5699 是旧的双臂 Piper 版，不能用")
    p.add_argument("--head-key", default="front", help="请求里头部图像的键名")
    p.add_argument("--wrist-key", default="wrist", help="请求里腕部图像的键名")
    p.add_argument("--image-layout", choices=["chw", "hwc"], default="chw")
    p.add_argument("--head-topic", default="/cam/head/color/image_raw")
    p.add_argument("--cam-impl", choices=["fast", "plugin"], default="fast",
                   help="fast：本脚本的轻量订阅（队列 1，不带深度）；plugin：录制插件的相机类")
    p.add_argument("--head-depth-topic", default="", help="仅 --cam-impl plugin")
    p.add_argument("--wrist-topic", default="/cam/wrist/color/image_raw")
    p.add_argument("--wrist-depth-topic", default="", help="仅 --cam-impl plugin")
    p.add_argument("--max-image-age-ms", type=float, default=500, help="多久没收到新画面就停止")
    p.add_argument("--settle-s", type=float, default=0.3, help="每块执行完、取观测前至少等这么久（另外还要等到停下后的新画面）")
    p.add_argument("--action-hz", type=float, default=10.0, help="动作块每步的频率（= 数据集 fps）")
    p.add_argument("--control-hz", type=float, default=30.0, help="下发给从臂的频率（插件按 30 Hz 设计）")
    p.add_argument("--exec-steps", type=int, default=10, help="每块执行前几步再重新推理（10 步 = 1 s）")
    p.add_argument("--slow", type=float, default=1.0, help="live 执行放慢倍数（3 = 用 3 倍时间走完）")
    p.add_argument("--max-chunks", type=int, default=50)
    p.add_argument("--max-duration-s", type=float, default=120)
    p.add_argument("--max-first-dev-deg", type=float, default=5.0, help="首帧离当前位置的上限（插件也是 5°）")
    p.add_argument("--max-jump-deg", type=float, default=30.0,
                   help="动作块内相邻两步（0.1 s）跳变的硬上限，超过就拒绝整块（明显不合理的输出）")
    p.add_argument("--max-joint-vel-deg", type=float, default=40.0,
                   help="实际执行时关节速度上限（°/s）：跳变大的步会自动拉长时间，而不是拒绝")
    p.add_argument("--dry-interval-s", type=float, default=1.0)
    p.add_argument("--infer-timeout-s", type=float, default=10.0, help="模型多久没返回算超时（手臂此时静止）")
    p.add_argument("--infer-retries", type=int, default=1, help="超时后重连重试几次，再超时就停止")
    p.add_argument("--fresh-after-stop-s", type=float, default=0.3,
                   help="只用手臂停下这么久之后才收到的画面（相机传输有 0.1~0.5 s 延迟）")
    p.add_argument("--piper-compat", action="store_true",
                   help="仅 dry：按当前双臂 Piper 服务的格式发请求，只测连通和耗时")
    p.add_argument("--log-dir", default="/data/panhe/vla_runs")
    p.add_argument("--step-file", default="", help="测试台写当前步骤的文件（如 ~/.nero_vla_step）")
    p.add_argument("--step-prompt", action="store_true", help="把 Step k/4 文本接到指令后（只给带步骤训练的模型用）")
    p.add_argument("--variant", default="", help="实验变体标签，记进 summary（E0 / E1 / M1 / M2 ...）")
    p.add_argument("--save-images", type=int, default=1, help="1 = 每次推理存头部/腕部图像")
    p.add_argument("--feedback-every", type=int, default=3,
                   help="执行时每 N 个控制周期读一次从臂实际位置（30 Hz 下 3 = 10 Hz；0 = 不记）")
    return p


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args = argument_parser().parse_args(argv)
    if args.exec_steps < 1 or args.control_hz < args.action_hz or args.slow < 1:
        raise SystemExit("--exec-steps 至少 1，--control-hz 不能小于 --action-hz，--slow 不能小于 1")
    if args.piper_compat and args.mode != "dry":
        raise SystemExit("--piper-compat 只允许在 dry 模式下用（输出没有物理意义）")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
