"""深度图单独存储：每条每个相机一个无损 16 位视频（FFV1，gray16le，单位毫米），帧号与 LeRobot 数据集逐帧对应。
目录：<数据集>/depth/<相机>/episode_000123.mkv，说明在 <数据集>/depth/depth_info.json。
读：frames = load_depth(root, episode, "front")  →  uint16 数组 (N, H, W)，单位毫米，0 = 无效。"""
import json, queue, shutil, threading, uuid
from pathlib import Path
import numpy as np

CODEC = dict(codec="ffv1", pix_fmt="gray16le", options={"level": "3", "g": "1", "slices": "4", "threads": "2"})


class _Writer:
    """后台线程编码，录制循环里只是把数组放进队列。"""

    def __init__(self, path: Path, fps: int, w: int, h: int):
        import av
        self.path, self.n, self.err = path, 0, None
        self.c = av.open(str(path), "w")
        self.s = self.c.add_stream(CODEC["codec"], rate=fps)
        self.s.width, self.s.height, self.s.pix_fmt, self.s.options = w, h, CODEC["pix_fmt"], dict(CODEC["options"])
        self.q: queue.Queue = queue.Queue(maxsize=600)
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start()

    def _run(self):
        import av
        while True:
            d = self.q.get()
            if d is None: break
            try:
                for pk in self.s.encode(av.VideoFrame.from_ndarray(d, format="gray16le")): self.c.mux(pk)
            except Exception as e:
                self.err = str(e)
        try:
            for pk in self.s.encode(): self.c.mux(pk)
            self.c.close()
        except Exception as e:
            self.err = str(e)

    def put(self, d):
        self.q.put(np.ascontiguousarray(d, dtype=np.uint16)); self.n += 1

    def close(self):
        self.q.put(None); self.t.join()


class DepthRecorder:
    def __init__(self, root, cams, fps):
        self.root, self.cams, self.fps = Path(root), list(cams), int(fps)
        self.tmp = self.root / "depth" / ".tmp"
        shutil.rmtree(self.tmp, ignore_errors=True)       # 上次异常退出留下的半截
        self.cur: dict[str, _Writer] = {}; self.dir = None

    def start(self):
        self.abort()
        self.dir = self.tmp / uuid.uuid4().hex; self.dir.mkdir(parents=True); self.cur = {}

    def add(self, depths: dict):
        """depths: {相机: uint16 (H, W) 或 None}。每个相机每帧都要写一帧；缺帧用全 0（无效）补，保证帧号对齐。"""
        if self.dir is None: self.start()
        for cam in self.cams:
            d = depths.get(cam)
            if cam not in self.cur:
                if d is None: continue
                self.cur[cam] = _Writer(self.dir / f"{cam}.mkv", self.fps, d.shape[1], d.shape[0])
            w = self.cur[cam]
            w.put(d if d is not None else np.zeros((w.s.height, w.s.width), np.uint16))

    def finish(self, episode: int, n_frames: int) -> dict:
        res = {}
        for cam, w in self.cur.items():
            w.close()
            dst = self.root / "depth" / cam / f"episode_{episode:06d}.mkv"; dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(w.path), dst)
            res[cam] = dict(frames=w.n, ok=w.err is None and w.n == n_frames, error=w.err)
        if self.dir: shutil.rmtree(self.dir, ignore_errors=True)
        self.cur, self.dir = {}, None
        info_p = self.root / "depth" / "depth_info.json"
        info = json.loads(info_p.read_text()) if info_p.exists() else dict(
            description="Depth aligned to color, uint16 millimetres (0 = invalid). One lossless FFV1 gray16le video per episode and camera; "
                        "frame i matches dataset frame_index i of the same episode.",
            fps=self.fps, unit="mm", codec="ffv1", pix_fmt="gray16le", path="depth/{camera}/episode_{episode_index:06d}.mkv", cameras=self.cams, episodes={})
        info["episodes"][str(episode)] = res
        info_p.write_text(json.dumps(info, indent=1))
        return res

    def abort(self):
        for w in self.cur.values():
            try: w.close()
            except Exception: pass
        if self.dir: shutil.rmtree(self.dir, ignore_errors=True)
        self.cur, self.dir = {}, None


def load_depth(root, episode: int, cam: str) -> np.ndarray:
    import av
    p = Path(root) / "depth" / cam / f"episode_{episode:06d}.mkv"
    with av.open(str(p)) as c:
        return np.stack([f.to_ndarray() for f in c.decode(video=0)])
