#!/usr/bin/env python3
"""只读：逐条核对一个数据集（帧数在 parquet / 视频 / 深度三处是否一致、10 Hz 时间戳、相机时间差、step、主臂夹爪、深度有效率）。
用法：python ~/nero_verify.py /data/panhe/lerobot_data/<名字>"""
import glob, json, sys
from pathlib import Path
import av, numpy as np, pandas as pd

R = Path(sys.argv[1]); info = json.loads((R / "meta/info.json").read_text()); fps = info["fps"]
df = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(R / "data/chunk-*/*.parquet")))], ignore_index=True)
ep = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(R / "meta/episodes/chunk-*/*.parquet")))], ignore_index=True)
cams = [k.split(".")[-1] for k in info["features"] if k.startswith("observation.images.")]
try:   # 失败数据集：step 到失败那一步为止，且每条都应标失败
    FAIL = any(t.get("dataset") == R.name and t.get("failure") for t in json.loads((Path.home() / "nero_tasks.json").read_text()).values())
except Exception:
    FAIL = False
print(f"数据集 {R.name}：v{info['codebase_version']}，{fps} Hz，{len(ep)} 条，{len(df)} 帧；字段 {len(info['features'])} 个")
vframes = {}
for cam in cams:
    for f in sorted(glob.glob(str(R / f"videos/observation.images.{cam}/chunk-*/*.mp4"))):
        with av.open(f) as c: vframes[(cam, Path(f).name)] = c.streams.video[0].frames or sum(1 for _ in c.decode(video=0))
bad = 0
for _, e in ep.iterrows():
    i = int(e["episode_index"]); d = df[df.episode_index == i]; n = len(d); msg = []
    ts = d["timestamp"].to_numpy(); ok_ts = np.allclose(np.diff(ts), 1 / fps, atol=1e-4)
    for cam in cams:
        t0, t1 = e[f"videos/observation.images.{cam}/from_timestamp"], e[f"videos/observation.images.{cam}/to_timestamp"]
        nv = round((t1 - t0) * fps)
        if nv != n: msg.append(f"{cam} 视频 {nv} 帧≠{n}")
        dp = R / "depth" / cam / f"episode_{i:06d}.mkv"
        if dp.exists():
            with av.open(str(dp)) as c: fr = [f.to_ndarray() for f in c.decode(video=0)]
            valid = np.mean([(x > 0).mean() for x in fr]) * 100
            if len(fr) != n: msg.append(f"{cam} 深度 {len(fr)} 帧≠{n}")
            dep = f"{cam} 深度 {len(fr)} 帧 有效像素 {valid:.0f}% 中位 {np.median(fr[len(fr) // 2][fr[len(fr) // 2] > 0]) if (fr[len(fr) // 2] > 0).any() else 0:.0f} mm"
        else:
            dep = f"{cam} 深度 缺"; msg.append(dep)
        print(f"  #{i} {dep}")
    sy = np.abs(np.hstack(d["camera.sync_ms"].to_numpy())) if "camera.sync_ms" in d else np.array([np.nan])
    st = np.hstack(d["step_index"].to_numpy()) if "step_index" in d else np.zeros(n, int)
    lg = np.hstack(d["leader.gripper.pos"].to_numpy()) * 1000 if "leader.gripper.pos" in d else np.array([np.nan])
    dr = np.abs(np.stack(d["camera.depth_rgb_ms"].to_numpy())[:, 0].astype(float)) if "camera.depth_rgb_ms" in d else None
    it = np.hstack(d["item_index"].to_numpy()) if "item_index" in d else None
    if it is not None and (len(set(it.tolist())) != 1 or it[0] == 0): msg.append(f"物品标签 {sorted(set(it.tolist()))}（0 = 未标）")
    steps = [int(k) for k, _ in __import__("itertools").groupby(st)]
    if dr is not None and (not np.isfinite(dr).all() or dr.max() > 20): msg.append(f"头部深度−RGB 时间差最大 {np.nanmax(dr):.1f} ms 或有缺值")
    if not np.isfinite(sy).any(): msg.append("没有相机时间差")
    elif np.nanmax(sy) > 20: msg.append("相机时间差 > 20 ms")
    if FAIL:
        if not steps or steps != list(range(1, len(steps) + 1)): msg.append(f"step 顺序 {steps}（失败数据要从 1 开始按顺序）")
        if str(e["episode_success"]).lower() in ("true", "success", "1"): msg.append("失败数据集里标成了成功")
    elif steps != [1, 2, 3, 4]: msg.append(f"step 顺序 {steps}")
    if not ok_ts: msg.append("时间戳不是均匀 10 Hz")
    if not np.isfinite(lg).all(): msg.append("主臂夹爪有缺值")
    bad += bool(msg)
    print(f"#{i} {e['episode_success']}" + (f" F{len(steps)}" if FAIL else "") + (f" 物品 {it[0]}" if it is not None else "") + f" {n} 帧 {n / fps:.1f} s | 时间戳{'均匀' if ok_ts else '不均匀'} | 相机时间差 最大 {np.nanmax(sy):.1f} ms 平均 {np.nanmean(sy):.1f} ms"
          f" | step {'→'.join(map(str, steps))} 各 {[int((st == k).sum()) for k in (1, 2, 3, 4)]} 帧 | 主臂夹爪 {np.nanmin(lg):.1f}–{np.nanmax(lg):.1f} mm"
          + (f" | 头部深度−RGB 最大 {np.nanmax(dr):.2f} ms" if dr is not None else " | 头部深度−RGB 未记录")
          + (f"  ✗ {'；'.join(msg)}" if msg else "  ✓"))
print(f"视频文件帧数：{vframes}")
print("全部通过" if not bad else f"{bad} 条有问题")
