import logging
import threading
import time
from functools import cached_property
from typing import Any

from lerobot.processor import RobotAction
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_nero_leader import NeroLeaderConfig

logger = logging.getLogger(__name__)

JOINT_KEYS = [f"joint{i}.pos" for i in range(1, 8)]


class NeroLeader(Teleoperator):
    """AgileX NERO 主臂。
    - 示教模式（leader，set_manual_control(True)）：只读 can1 上的示教帧，输出限位 + 限幅后的动作。
    - 跟随模式（follower，set_manual_control(False) / send_feedback）：主臂用 move_js 跟随给定关节角，
      用于模型执行时让主臂与从臂同步，按 i 接管时不跳。夹爪暂不跟随。
    """

    config_class = NeroLeaderConfig
    name = "nero_leader"

    def __init__(self, config: NeroLeaderConfig):
        super().__init__(config)
        self.config = config
        self._arm = None
        self._gripper = None
        self._connected = False
        self._last_ts = None
        self._last_change = 0.0
        self._out = None
        self._out_t = 0.0
        self._manual: bool | None = None   # True 示教，False 跟随，None 切换中/未知
        self._fb_cmd: list[float] | None = None
        self._fb_t = 0.0
        self._last_grip: float | None = None
        self._node = self._executor = self._spin = None
        self._pos_on = False
        self._pos_active = False
        self._fol: list[float] | None = None
        self._fol_t = 0.0

    # ---------- 特征 ----------
    @cached_property
    def action_features(self) -> dict[str, type]:
        keys = list(JOINT_KEYS) + (["gripper.pos"] if self.config.use_gripper else [])
        return dict.fromkeys(keys, float)

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return dict.fromkeys(JOINT_KEYS, float)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ---------- 连接 ----------
    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        from pyAgxArm import AgxArmFactory, NeroFW, create_agx_arm_config

        fw = getattr(NeroFW, self.config.firmware_version)
        arm = AgxArmFactory.create_arm(
            create_agx_arm_config(robot="nero", comm="can", channel=self.config.channel, firmeware_version=fw)
        )
        arm.connect()
        try:
            if self.config.ensure_leader_mode_on_connect:
                arm.set_leader_mode()
                time.sleep(self.config.mode_switch_settle_s)
            grip = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER) if self.config.use_gripper else None
            deadline = time.monotonic() + self.config.connect_timeout_s
            while True:
                la = arm.get_leader_joint_angles()
                grip_ok = grip is None or grip.get_gripper_ctrl_states() is not None
                if la is not None and grip_ok:
                    break
                if time.monotonic() > deadline:
                    raise ConnectionError(
                        f"{self.config.connect_timeout_s}s 内没读到 {self.config.channel} 上的主臂示教数据："
                        "主臂上电了吗？在示教模式吗？"
                    )
                time.sleep(0.02)
        except Exception:
            arm.disconnect()
            raise

        self._arm, self._gripper = arm, grip
        self._last_ts = None
        self._last_change = time.monotonic()
        self._out = None
        self._fb_cmd = None
        self._manual = True
        self._connected = True
        if self.config.positioning_follow:
            self._start_ros()
        logger.info("%s connected (%s, 示教模式).", self, self.config.channel)

    # ---------- 工具 ----------
    def _clamp(self, i: int, v: float) -> float:
        lo, hi = self.config.joint_limits[i]
        return min(max(float(v), lo), hi)

    def _read_master_joints(self) -> list[float]:
        """跟随模式下读主臂自身的关节反馈。"""
        ja = self._arm.get_joint_angles()
        if ja is None:
            raise ConnectionError("读不到主臂关节反馈（跟随模式）")
        vals = [float(v) for v in ja.msg]
        if len(vals) < 7:
            raise ConnectionError(f"主臂关节反馈只有 {len(vals)} 个值")
        return vals[:7]

    def _read_teach_action(self, now: float) -> RobotAction:
        la = self._arm.get_leader_joint_angles()
        if la is None:
            raise ConnectionError("读不到主臂示教数据")
        if la.timestamp != self._last_ts:
            self._last_ts = la.timestamp
            self._last_change = now
        elif now - self._last_change > self.config.max_data_age_s:
            raise ConnectionError(f"主臂示教数据 {now - self._last_change:.2f}s 未更新，检查 can1 和主臂电源")
        action: RobotAction = {k: self._clamp(i, la.msg[i]) for i, k in enumerate(JOINT_KEYS)}
        if self.config.use_gripper:
            g = self._gripper.get_gripper_ctrl_states()
            if g is None:
                raise ConnectionError("读不到主臂夹爪")
            width = float(g.msg.value) / self.config.leader_gripper_max_m * self.config.follower_gripper_max_m
            action["gripper.pos"] = min(max(width, 0.0), self.config.follower_gripper_max_m)
        return action

    # ---------- 定位模式用的 ROS 订阅 ----------
    def _start_ros(self) -> None:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile
        from rclpy.signals import SignalHandlerOptions
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool

        if not rclpy.ok():
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self._node = Node(f"lerobot_nero_leader_{self.id or 'leader'}")
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._node.create_subscription(Bool, "/nero/positioning_state", self._on_pos, latched)
        self._node.create_subscription(JointState, self.config.follower_state_topic, self._on_fol, 10)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin.start()

    def _on_pos(self, msg) -> None:
        self._pos_on = bool(msg.data)

    def _on_fol(self, msg) -> None:
        v = dict(zip(msg.name, msg.position))
        if all(f"joint{i}" in v for i in range(1, 8)):
            self._fol = [float(v[f"joint{i}"]) for i in range(1, 8)]
            self._fol_t = time.monotonic()

    def _stop_ros(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
        if self._spin is not None:
            self._spin.join(timeout=2.0)
        if self._node is not None:
            self._node.destroy_node()
        self._node = self._executor = self._spin = None

    # ---------- 读 ----------
    @check_if_not_connected
    def get_action(self) -> RobotAction:
        now = time.monotonic()
        # 定位模式：主臂跟随从臂，输出从臂当前位置
        if self.config.positioning_follow and self._pos_on:
            if self._fol is None or now - self._fol_t > self.config.max_data_age_s:
                raise ConnectionError("定位模式下读不到从臂状态")
            fb = {k: self._fol[i] for i, k in enumerate(JOINT_KEYS)}
            if not self._pos_active:
                logger.warning("%s 进入定位模式：主臂切到跟随", self)
                self._pos_active = True
            self.send_feedback(fb)
            action: RobotAction = dict(fb)
            if self.config.use_gripper:
                action["gripper.pos"] = self._last_grip if self._last_grip is not None else 0.0
            return action
        if self._pos_active:
            self._pos_active = False
            self.set_manual_control(True)
            logger.warning("%s 退出定位模式：主臂切回示教", self)
        if self._manual is False:
            # 跟随模式：主臂不再广播示教帧，返回主臂自身位置
            joints = self._read_master_joints()
            action: RobotAction = {k: self._clamp(i, joints[i]) for i, k in enumerate(JOINT_KEYS)}
            if self.config.use_gripper:
                action["gripper.pos"] = self._last_grip if self._last_grip is not None else 0.0
            return action

        action = self._read_teach_action(now)
        # 单步限幅：输出的是平滑后的、从臂执行得了的指令
        if self._out is None or now - self._out_t > self.config.reseed_s:
            self._out = dict(action)
        else:
            for key in JOINT_KEYS:
                prev, step = self._out[key], self.config.max_step_rad
                self._out[key] = min(max(action[key], prev - step), prev + step)
            if "gripper.pos" in action:
                prev = self._out.get("gripper.pos", action["gripper.pos"])
                step = self.config.max_gripper_step_m
                self._out["gripper.pos"] = min(max(action["gripper.pos"], prev - step), prev + step)
        self._out_t = now
        self._last_grip = self._out.get("gripper.pos")
        return dict(self._out)

    # ---------- 模式切换 ----------
    def set_manual_control(self, enabled: bool) -> None:
        if not self._connected:
            raise RuntimeError(f"{self} 未连接")
        if enabled == self._manual:
            return
        cfg = self.config
        if enabled:
            self._arm.set_leader_mode()
            time.sleep(cfg.mode_switch_settle_s)
            self._out = None
            self._last_ts = None
            self._last_change = time.monotonic()
            self._fb_cmd = None
            self._manual = True
            logger.info("%s 切回示教模式（leader）", self)
            return

        self._manual = None  # 切换中
        self._arm.set_follower_mode()
        time.sleep(cfg.mode_switch_settle_s)
        if not self._arm.enable(timeout=cfg.enable_timeout_s):
            self._arm.set_leader_mode()
            self._manual = True
            raise RuntimeError("主臂切到跟随模式后未能使能，已切回示教模式")
        self._fb_cmd = self._read_master_joints()
        self._fb_t = time.monotonic()
        self._manual = False
        logger.info("%s 切到跟随模式（follower）", self)

    # ---------- 主臂跟随 ----------
    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        self.set_manual_control(False)
        if not all(k in feedback for k in JOINT_KEYS):
            return
        now = time.monotonic()
        if self._fb_cmd is None or now - self._fb_t > self.config.reseed_s:
            self._fb_cmd = self._read_master_joints()
        step = self.config.feedback_max_step_rad
        targets = []
        for i, k in enumerate(JOINT_KEYS):
            prev = self._fb_cmd[i]
            lo, hi = self.config.joint_limits[i]
            t = min(max(float(feedback[k]), min(lo, prev)), max(hi, prev))  # 已超限只许往里走
            t = min(max(t, prev - step), prev + step)
            targets.append(t)
        self._arm.move_js(targets)
        self._fb_cmd = targets
        self._fb_t = now

    # ---------- 断开 ----------
    @check_if_not_connected
    def disconnect(self) -> None:
        try:
            if self._manual is not True:
                try:
                    self._arm.set_leader_mode()
                    logger.info("%s 断开前已切回示教模式", self)
                except Exception as e:
                    logger.error("%s 切回示教模式失败：%s", self, e)
        finally:
            self._stop_ros()
            self._arm.disconnect()
            self._arm = self._gripper = None
            self._connected = False
            self._manual = None
            logger.info("%s disconnected.", self)
