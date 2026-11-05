import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import ColorMode

from .config_ros_image import RosImageCameraConfig

logger = logging.getLogger(__name__)


@dataclass
class CamFrame:
    stamp_ms: float                 # ROS header 时间戳（系统时钟，毫秒）
    color: np.ndarray
    depth: np.ndarray | None        # uint16 毫米，已对齐到彩色；没订阅深度时为 None
    recv: float                     # 收到的时刻（monotonic 秒）
    depth_stamp_ms: float | None = None   # 配对到的深度帧时间戳


class RosImageCamera(Camera):
    """把 ROS 2 图像话题当作 LeRobot 相机。相机本身由 realsense2_camera 等驱动独占，网页等其他程序可同时订阅。"""

    def __init__(self, config: RosImageCameraConfig):
        super().__init__(config)
        self.config = config
        self._node = self._executor = self._thread = None
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self._seq = 0
        self._last_read_seq = 0
        self._error: str | None = None
        self._connected = False
        self._buf: deque[CamFrame] = deque(maxlen=config.buffer_len)
        self._pending_c: dict[float, tuple] = {}
        self._pending_d: dict[float, np.ndarray] = {}
        self.last: CamFrame | None = None   # 最近一次 async_read / read_nearest 返回的那一帧

    def __str__(self) -> str:
        return f"RosImageCamera({self.config.topic})"

    @property
    def is_connected(self) -> bool:
        return self._connected

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        return []

    # ---------- 连接 ----------
    def connect(self, warmup: bool = True) -> None:
        if self._connected:
            raise RuntimeError(f"{self} 已经连接")
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from rclpy.signals import SignalHandlerOptions
        from sensor_msgs.msg import Image

        if not rclpy.ok():
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        name = "lerobot_cam_" + "".join(c if c.isalnum() else "_" for c in self.config.topic).strip("_")
        self._node = Node(name)
        self._node.create_subscription(Image, self.config.topic, self._on_image, 10)  # 可靠模式：大图像用 best effort 会大量丢帧（实测头部彩色 5.6 → 15 fps）
        if self.config.depth_topic:
            self._node.create_subscription(Image, self.config.depth_topic, self._on_depth, 10)  # 可靠模式：大图像用 best effort 会大量丢帧（实测头部彩色 5.6 → 15 fps）
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

        deadline = time.monotonic() + self.config.connect_timeout_s
        with self._cond:
            while self._frame is None and time.monotonic() < deadline:
                self._cond.wait(max(deadline - time.monotonic(), 0.01))
            got = self._frame is not None
        if not got:
            err = self._error
            self._shutdown()
            raise ConnectionError(
                f"{self.config.connect_timeout_s}s 内没收到可用画面 {self.config.topic}"
                + (f"：{err}" if err else "：相机驱动在运行吗？")
            )
        if warmup and self.config.warmup_s > 0:
            time.sleep(self.config.warmup_s)
        self._connected = True
        logger.info("%s connected.", self)

    def _decode(self, msg) -> np.ndarray:
        enc = msg.encoding.lower()
        if enc not in ("rgb8", "bgr8"):
            raise ValueError(f"不支持的编码 {msg.encoding}（只支持 rgb8/bgr8）")
        if msg.width != self.config.width or msg.height != self.config.height:
            raise ValueError(
                f"画面 {msg.width}x{msg.height} 与配置 {self.config.width}x{self.config.height} 不符"
            )
        buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        img = buf[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
        if (enc == "rgb8") != (self.config.color_mode == ColorMode.RGB):
            img = img[:, :, ::-1]
        return np.array(img)  # 拷贝成独立、可写的数组

    @staticmethod
    def _stamp_ms(msg) -> float:
        return msg.header.stamp.sec * 1000.0 + msg.header.stamp.nanosec / 1e6

    def _push(self, st: float, color: np.ndarray, depth, dst=None) -> None:
        with self._cond:
            fr = CamFrame(st, color, depth, time.monotonic(), dst)
            self._buf.append(fr)
            self._frame = color
            self._stamp = fr.recv
            self._seq += 1
            self._cond.notify_all()

    def _try_pair(self) -> None:
        tol = self.config.pair_tolerance_ms
        for sc in sorted(self._pending_c):
            sd = min(self._pending_d, key=lambda x: abs(x - sc), default=None)
            if sd is not None and abs(sd - sc) <= tol:
                color = self._pending_c.pop(sc)[0]; depth = self._pending_d.pop(sd)
                for k in [k for k in self._pending_c if k < sc]: self._pending_c.pop(k)
                for k in [k for k in self._pending_d if k < sd]: self._pending_d.pop(k)
                self._push(sc, color, depth, sd)
        while len(self._pending_c) > 10: self._pending_c.pop(min(self._pending_c))
        while len(self._pending_d) > 10: self._pending_d.pop(min(self._pending_d))

    def _on_image(self, msg) -> None:
        try:
            frame = self._decode(msg)
        except Exception as e:
            if self._error != str(e):
                logger.error("%s: %s", self, e)
            self._error = str(e)
            return
        st = self._stamp_ms(msg)
        if not self.config.depth_topic:
            self._push(st, frame, None); return
        with self._cond:
            self._pending_c[st] = (frame,)
            self._try_pair()

    def _on_depth(self, msg) -> None:
        if msg.encoding.lower() not in ("16uc1", "mono16") or msg.width != self.config.width or msg.height != self.config.height:
            e = f"深度图 {msg.encoding} {msg.width}x{msg.height} 不符合（要 16UC1 {self.config.width}x{self.config.height}，对齐到彩色）"
            if self._error != e: logger.error("%s: %s", self, e)
            self._error = e; return
        d = np.array(np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.step // 2)[:, : msg.width])
        with self._cond:
            self._pending_d[self._stamp_ms(msg)] = d
            self._try_pair()

    # ---------- 读 ----------
    def read(self, color_mode: ColorMode | None = None) -> np.ndarray:
        """阻塞到下一张新画面。"""
        if not self._connected:
            raise RuntimeError(f"{self} 未连接")
        with self._cond:
            seq0 = self._last_read_seq
            if not self._cond.wait_for(lambda: self._seq > seq0, timeout=self.config.read_timeout_ms / 1000):
                raise TimeoutError(f"{self} {self.config.read_timeout_ms} ms 内没有新画面")
            self._last_read_seq = self._seq
            return self._frame

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        """返回最新画面；只要不超过 max_age_ms 就直接返回（低帧率相机会重复返回同一张）。"""
        if not self._connected:
            raise RuntimeError(f"{self} 未连接")
        max_age = self.config.max_age_ms / 1000

        def fresh():
            return self._frame is not None and time.monotonic() - self._stamp <= max_age

        with self._cond:
            if not fresh() and not self._cond.wait_for(fresh, timeout=timeout_ms / 1000):
                raise TimeoutError(f"{self} 超过 {self.config.max_age_ms} ms 没有新画面，检查相机驱动")
            self.last = self._buf[-1]
            return self._frame

    def latest(self) -> CamFrame:
        """最新一帧（含时间戳和深度），并检查是否断流。"""
        self.async_read()
        return self.last

    def recent(self, n: int = 3) -> list:
        """最近 n 帧（新的在前），用于在几帧里挑配对最好的一组。"""
        self.async_read()
        with self._cond:
            return list(self._buf)[-n:][::-1]

    def read_nearest(self, stamp_ms: float) -> CamFrame:
        """缓存里时间戳最接近 stamp_ms 的一帧（用于两个相机按时间戳配对）。"""
        self.async_read()
        with self._cond:
            fr = min(self._buf, key=lambda f: abs(f.stamp_ms - stamp_ms))
        self.last = fr
        return fr

    # ---------- 断开 ----------
    def disconnect(self) -> None:
        self._shutdown()
        self._connected = False
        logger.info("%s disconnected.", self)

    def _shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._node is not None:
            self._node.destroy_node()
        self._executor = self._node = self._thread = None
