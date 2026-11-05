import logging
import math
import threading
import time
from functools import cached_property

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots import Robot
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_nero_follower import NeroFollowerConfig

logger = logging.getLogger(__name__)

JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
JOINT_KEYS = [f"{n}.pos" for n in JOINT_NAMES]


class NeroFollower(Robot):
    """AgileX NERO 7 轴从臂 + AGX 夹爪，经 ROS 2 驱动通信。"""

    config_class = NeroFollowerConfig
    name = "nero_follower"

    def __init__(self, config: NeroFollowerConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)

        self._node = self._executor = self._spin_thread = None
        self._pub_arm = self._pub_grip = self._gate = self._hold = None
        self._lock = threading.Lock()
        self._latest: dict[str, float] | None = None
        self._latest_time = 0.0
        self._connected = False

        self._cmd: list[float] | None = None
        self._gcmd: float | None = None
        self._cmd_time = 0.0
        self._clip_frames = 0
        self._clip_j = 0
        self._clip_g = 0
        self._max_j_deg = 0.0
        self._max_g_mm = 0.0
        self._last_clip_log = 0.0
        self._positioning = False
        self._pos_pub = self._pos_srv = None
        self.cam_frames: dict = {}

    # ---------- 特征 ----------
    @property
    def _motor_keys(self) -> list[str]:
        return JOINT_KEYS + (["gripper.pos"] if self.config.use_gripper else [])

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {k: (c.height, c.width, 3) for k, c in self.config.cameras.items()}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**dict.fromkeys(self._motor_keys, float), **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(self._motor_keys, float)

    @property
    def is_connected(self) -> bool:
        return self._connected and all(c.is_connected for c in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ---------- ROS 工具 ----------
    def _call(self, client, request, timeout_s: float = 2.0):
        if client is None or not client.wait_for_service(timeout_sec=timeout_s):
            return None
        future = client.call_async(request)
        deadline = time.monotonic() + timeout_s
        while not future.done():
            if time.monotonic() > deadline:
                return None
            time.sleep(0.01)
        return future.result()

    def _set_gate(self, enabled: bool) -> bool:
        from std_srvs.srv import SetBool

        req = SetBool.Request()
        req.data = enabled
        res = self._call(self._gate, req)
        ok = bool(res and res.success)
        logger.info("控制锁 %s: %s", "打开" if enabled else "关闭", res.message if res else "无响应")
        return ok

    def _hold_in_place(self) -> None:
        from std_srvs.srv import Empty

        if self._call(self._hold, Empty.Request()) is None:
            logger.error("原地保持(/emergency_stop)无响应！请用急停窗口或实体急停。")

    # ---------- 连接 ----------
    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.signals import SignalHandlerOptions
        from sensor_msgs.msg import JointState
        from std_srvs.srv import Empty, SetBool

        if not rclpy.ok():
            # 不让 ROS 接管 Ctrl+C，保证 disconnect 时还能关锁、保持
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self._node = Node(f"lerobot_nero_{self.id or 'follower'}")
        self._node.create_subscription(JointState, self.config.state_topic, self._on_state, 10)
        self._pub_arm = self._node.create_publisher(JointState, self.config.command_topic, 1)
        self._pub_grip = self._node.create_publisher(JointState, self.config.gripper_topic, 1)
        self._gate = self._node.create_client(SetBool, "/control_enable")
        self._hold = self._node.create_client(Empty, "/emergency_stop")
        from rclpy.qos import DurabilityPolicy, QoSProfile
        from std_msgs.msg import Bool
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._pos_pub = self._node.create_publisher(Bool, "/nero/positioning_state", latched)
        self._pos_srv = self._node.create_service(SetBool, "/nero/positioning", self._on_positioning)
        self._publish_positioning()
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        try:
            deadline = time.monotonic() + self.config.connect_timeout_s
            while True:
                with self._lock:
                    ready = self._latest is not None and (
                        not self.config.use_gripper or "gripper" in self._latest
                    )
                if ready:
                    break
                if time.monotonic() > deadline:
                    raise ConnectionError(
                        f"{self.config.connect_timeout_s}s 内没收到 {self.config.state_topic}"
                        f"{'（含 gripper）' if self.config.use_gripper else ''}：驱动在运行吗？"
                        "启用夹爪需 effector_type:=agx_gripper"
                    )
                time.sleep(0.02)

            if not self.config.read_only and self.config.manage_control_gate:
                if not self._set_gate(True):
                    raise ConnectionError("打开控制锁失败（/control_enable 无响应或被拒绝）")

            for cam in self.cameras.values():
                cam.connect()
        except Exception:
            if not self.config.read_only and self.config.manage_control_gate and self._gate is not None:
                self._set_gate(False)
            self._shutdown_ros()
            raise

        self._cmd = None
        self._connected = True
        logger.info("%s connected (%s).", self, "只读" if self.config.read_only else "可控制")

    def _on_positioning(self, req, res):
        self._positioning = bool(req.data)
        self._cmd = None  # 退出定位后重新做对齐检查
        self._publish_positioning()
        res.success = True
        res.message = "定位模式开启：从臂不再执行主臂指令" if req.data else "定位模式关闭：从臂恢复跟随主臂"
        logger.warning(res.message)
        return res

    def _publish_positioning(self) -> None:
        from std_msgs.msg import Bool

        if self._pos_pub is not None:
            m = Bool()
            m.data = self._positioning
            self._pos_pub.publish(m)

    def _on_state(self, msg) -> None:
        values = dict(zip(msg.name, msg.position))
        if not all(n in values for n in JOINT_NAMES):
            return
        state = {n: float(values[n]) for n in JOINT_NAMES}
        if "gripper" in values:
            state["gripper"] = float(values["gripper"])
        with self._lock:
            self._latest = state
            self._latest_time = time.monotonic()

    def _fresh_state(self) -> dict[str, float]:
        with self._lock:
            state, t = self._latest, self._latest_time
        age = time.monotonic() - t
        if state is None or age > self.config.max_state_age_s:
            raise ConnectionError(f"从臂状态已过期（{age:.2f}s），检查驱动和 can0")
        return state

    # ---------- 读 ----------
    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        state = self._fresh_state()
        obs: RobotObservation = {f"{n}.pos": state[n] for n in JOINT_NAMES}
        if self.config.use_gripper:
            obs["gripper.pos"] = state.get("gripper", 0.0)
        ref = self.config.camera_sync_ref
        if ref and ref in self.cameras and all(hasattr(c, "read_nearest") for c in self.cameras.values()):
            # 头部最近 3 帧里，挑和腕部配得最好的一帧：优先最新的，只要所有相机都在 18 ms 内（标准 20 ms）；否则取最大差最小的一组
            others = {k: c for k, c in self.cameras.items() if k != ref}
            best = None
            for r in self.cameras[ref].recent(3):
                cand = {ref: r, **{k: c.read_nearest(r.stamp_ms) for k, c in others.items()}}
                worst = max([abs(f.stamp_ms - r.stamp_ms) for f in cand.values()] or [0.0])
                if best is None or worst < best[0] - 1e-6: best = (worst, cand)
                if worst <= 18.0: best = (worst, cand); break   # 留 2 ms 余量
            frames = best[1]
            for key, fr in frames.items():
                obs[key] = fr.color
            self.cam_frames = frames   # 给录制包装读时间戳和深度
        else:
            for key, cam in self.cameras.items():
                obs[key] = cam.async_read()
        return obs

    # ---------- 写 ----------
    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if not all(k in action for k in JOINT_KEYS):
            logger.debug("动作缺少关节，忽略：%s", list(action))
            return {}

        cur = self._fresh_state()
        now = time.monotonic()
        cfg = self.config

        # 定位模式：不发任何指令，返回从臂当前位置（网页等外部程序在移动从臂）
        if self._positioning:
            self._cmd = None
            held: RobotAction = {f"{n}.pos": cur[n] for n in JOINT_NAMES}
            if cfg.use_gripper:
                held["gripper.pos"] = cur.get("gripper", 0.0)
            return held

        # 开始或中断后恢复：检查主从是否对齐，以当前实际位置为起点
        if self._cmd is None or now - self._cmd_time > cfg.command_reseed_s:
            err = max(abs(float(action[f"{n}.pos"]) - cur[n]) for n in JOINT_NAMES)
            if err > cfg.align_tolerance_rad:
                raise RuntimeError(
                    f"主从未对齐：最大关节差 {math.degrees(err):.1f}° > {math.degrees(cfg.align_tolerance_rad):.0f}°。"
                    "请先运行 align_follower.py"
                )
            self._cmd = [cur[n] for n in JOINT_NAMES]
            self._gcmd = cur.get("gripper")

        # 跟踪误差：上一次指令与实际位置相差过大，说明从臂受阻或异常
        if not cfg.read_only:
            track = max(abs(self._cmd[i] - cur[n]) for i, n in enumerate(JOINT_NAMES))
            if track > cfg.max_tracking_error_rad:
                if cfg.manage_control_gate:
                    self._set_gate(False)
                self._hold_in_place()
                raise RuntimeError(f"从臂跟踪误差 {math.degrees(track):.1f}° 过大，已关锁并原地保持")

        clipped = False
        joint_clipped = False
        targets = []
        for i, n in enumerate(JOINT_NAMES):
            goal = float(action[f"{n}.pos"])
            prev = self._cmd[i]
            lo, hi = cfg.joint_limits[i]
            t = min(max(goal, min(lo, prev)), max(hi, prev))   # 已超限时只允许往限位内走
            t = min(max(t, prev - cfg.max_step_rad), prev + cfg.max_step_rad)
            if abs(t - goal) > 1e-6:
                clipped = joint_clipped = True
                self._max_j_deg = max(self._max_j_deg, math.degrees(abs(t - goal)))
            targets.append(t)
        self._cmd = targets
        self._cmd_time = now
        sent: RobotAction = dict(zip(JOINT_KEYS, targets))

        grip_target = None
        if cfg.use_gripper and "gripper.pos" in action:
            g = min(max(float(action["gripper.pos"]), 0.0), cfg.gripper_max_m)
            prev_g = self._gcmd if self._gcmd is not None else g
            grip_target = min(max(g, prev_g - cfg.max_gripper_step_m), prev_g + cfg.max_gripper_step_m)
            gd = abs(grip_target - float(action["gripper.pos"]))
            if gd > 1e-6:
                clipped = True
                self._clip_g += 1
                self._max_g_mm = max(self._max_g_mm, gd * 1000)
            self._gcmd = grip_target
            sent["gripper.pos"] = grip_target

        if joint_clipped:
            self._clip_j += 1
        if clipped:
            self._clip_frames += 1
            if now - self._last_clip_log > 1.0:
                logger.warning(
                    "限幅累计 %d 帧（关节 %d 帧，近 1 s 最大差 %.1f°；夹爪 %d 帧，近 1 s 最大差 %.1f mm）",
                    self._clip_frames, self._clip_j, self._max_j_deg, self._clip_g, self._max_g_mm,
                )
                self._last_clip_log = now
                self._max_j_deg = self._max_g_mm = 0.0

        if cfg.read_only:
            return sent

        from sensor_msgs.msg import JointState

        stamp = self._node.get_clock().now().to_msg()
        m = JointState()
        m.header.stamp = stamp
        m.name = list(JOINT_NAMES)
        m.position = targets
        self._pub_arm.publish(m)
        if grip_target is not None:
            g = JointState()
            g.header.stamp = stamp
            g.name = ["gripper"]
            g.position = [grip_target]
            g.effort = [cfg.gripper_force_n]
            self._pub_grip.publish(g)
        return sent

    # ---------- 断开 ----------
    @check_if_not_connected
    def disconnect(self) -> None:
        try:
            if not self.config.read_only and self.config.manage_control_gate:
                self._set_gate(False)
                self._hold_in_place()
            for cam in self.cameras.values():
                cam.disconnect()
        finally:
            if self._clip_frames:
                logger.warning("本次共 %d 帧被限幅：关节 %d 帧，夹爪 %d 帧", self._clip_frames, self._clip_j, self._clip_g)
            self._shutdown_ros()
            self._connected = False
            logger.info("%s disconnected.", self)

    def _shutdown_ros(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=2.0)
        if self._node is not None:
            self._node.destroy_node()
        self._executor = self._node = self._spin_thread = None
        self._pub_arm = self._pub_grip = self._gate = self._hold = None
