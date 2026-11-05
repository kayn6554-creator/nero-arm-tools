from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig, ColorMode


@CameraConfig.register_subclass("ros_image")
@dataclass
class RosImageCameraConfig(CameraConfig):
    """订阅一个 ROS 2 图像话题（sensor_msgs/Image，rgb8 或 bgr8）。
    fps / width / height 必须与发布端一致：width/height 不符会直接报错，fps 用于数据集元信息。"""

    topic: str = "/cam/head/color/image_raw"
    color_mode: ColorMode = ColorMode.RGB
    max_age_ms: int = 500          # async_read：最新画面超过这个时间就算断流
    read_timeout_ms: int = 1000    # read：等新画面的最长时间
    connect_timeout_s: float = 5.0
    warmup_s: float = 1.0
    # v2：同时订阅对齐到彩色的深度图（16UC1，毫米），按 ROS 时间戳与彩色配对；缓存最近若干帧供跨相机按时间戳配对
    depth_topic: str | None = None
    pair_tolerance_ms: float = 3.0
    buffer_len: int = 45

    def __post_init__(self):
        if self.fps is None or self.width is None or self.height is None:
            raise ValueError("ros_image 相机必须指定 fps、width、height（与 ROS 发布端一致）")
