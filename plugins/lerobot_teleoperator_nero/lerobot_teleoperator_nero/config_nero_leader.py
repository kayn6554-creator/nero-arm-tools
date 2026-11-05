import math
from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig


def nero_joint_limits(margin_deg: float = 1.0) -> list[tuple[float, float]]:
    """SDK 的 NERO 限位，j7 按急停实测收紧到 ±80°，两端各留 margin_deg。
    必须与从臂插件 lerobot_robot_nero 的限位保持一致。"""
    lim = [(-155, 155), (-100, 100), (-158, 158), (-58, 123), (-158, 158), (-42, 55), (-80, 80)]
    return [(math.radians(a + margin_deg), math.radians(b - margin_deg)) for a, b in lim]


@TeleoperatorConfig.register_subclass("nero_leader")
@dataclass
class NeroLeaderConfig(TeleoperatorConfig):
    """NERO 主臂。示教模式（leader）只读 can1；跟随模式（follower）用 move_js 跟随给定关节角。"""

    channel: str = "can1"
    firmware_version: str = "V112"

    use_gripper: bool = True
    leader_gripper_max_m: float = 0.0105
    follower_gripper_max_m: float = 0.100

    max_data_age_s: float = 0.5
    connect_timeout_s: float = 3.0

    # 示教模式输出的单步限幅（与从臂插件一致）：数据集里的动作 = 从臂实际执行的指令
    max_step_rad: float = 0.035        # 30 fps 约 60°/s
    max_gripper_step_m: float = 0.007  # 30 fps 约 210 mm/s
    reseed_s: float = 0.5

    # ---- 第二期：主臂跟随（send_feedback）----
    ensure_leader_mode_on_connect: bool = True   # 连接时先切到示教模式
    feedback_max_step_rad: float = 0.02          # 主臂跟随单步限幅，30 fps 约 34°/s
    mode_switch_settle_s: float = 0.3            # 切换模式后的等待时间
    enable_timeout_s: float = 3.0

    # ---- 定位模式：从臂插件的 /nero/positioning 开启时，主臂自动跟随从臂 ----
    positioning_follow: bool = True
    follower_state_topic: str = "/feedback/joint_states"

    joint_limits: list[tuple[float, float]] = field(default_factory=nero_joint_limits)
