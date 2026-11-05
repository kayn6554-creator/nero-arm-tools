import math
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


def nero_joint_limits(margin_deg: float = 1.0) -> list[tuple[float, float]]:
    """SDK 的 NERO 限位，j7 按急停实测收紧到 ±80°，两端各留 margin_deg。
    必须与主臂插件 lerobot_teleoperator_nero 的限位保持一致。"""
    lim = [(-155, 155), (-100, 100), (-158, 158), (-58, 123), (-158, 158), (-42, 55), (-80, 80)]
    return [(math.radians(a + margin_deg), math.radians(b - margin_deg)) for a, b in lim]


@RobotConfig.register_subclass("nero_follower")
@dataclass
class NeroFollowerConfig(RobotConfig):
    """NERO 从臂：经学长的 agx_arm ROS 2 驱动收发，不直接碰 CAN。"""

    state_topic: str = "/feedback/joint_states"
    command_topic: str = "/control/move_js"       # 关节：瞬时指令，不做平滑
    gripper_topic: str = "/control/joint_states"  # 夹爪：只发 name=["gripper"]

    # 安全：默认只读，True 时 send_action 只计算不发送
    read_only: bool = True
    # 非只读时：connect 打开控制锁，disconnect 关锁并原地保持
    manage_control_gate: bool = True

    use_gripper: bool = True
    gripper_max_m: float = 0.100
    gripper_force_n: float = 1.0

    # 单步限幅（相对上一次指令），只做兜底；平滑在主臂插件里做。30 fps：0.035 rad ≈ 60°/s
    max_step_rad: float = 0.035
    max_gripper_step_m: float = 0.007

    # 开始或中断后恢复时，主从差值超过它就拒绝执行
    align_tolerance_rad: float = math.radians(5.0)
    # 发出的指令与从臂实际位置差超过它：关锁、保持、报错（实测 57°/s 时最大 9.7°）
    max_tracking_error_rad: float = math.radians(15.0)
    # 超过这么久没收到动作，视为中断，下一次重新做对齐检查
    command_reseed_s: float = 0.5

    max_state_age_s: float = 0.5
    connect_timeout_s: float = 3.0

    joint_limits: list[tuple[float, float]] = field(default_factory=nero_joint_limits)

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # 多相机按时间戳配对：以这个相机的最新一帧为准，其他相机取时间戳最接近的一帧（需要 ros_image 相机）。None = 各取最新
    camera_sync_ref: str | None = None
