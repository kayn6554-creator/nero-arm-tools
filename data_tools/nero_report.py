#!/usr/bin/env python3
"""离线生成/刷新报告：python ~/nero_report.py /data/panhe/lerobot_data/<名字>
从 parquet 重新计算每条的统计；录制时留下的截图和大阶段会保留。"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nero_report_lib as NR

root = Path(sys.argv[1]).resolve(); rep = root / "report"
fps = json.loads((root / "meta/info.json").read_text())["fps"]
if not glob.glob(str(root / "meta/episodes/chunk-*/*.parquet")): sys.exit("还没有保存任何一条，跳过报告")
ep = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(root / "meta/episodes/chunk-*/*.parquet")))], ignore_index=True)
files = sorted(glob.glob(str(root / "data/chunk-*/*.parquet")))
cols = ["episode_index", "action", "observation.state", "complementary_info.is_intervention"]
import pyarrow.parquet as pq
has_step = NR.STEP_KEY in pq.read_schema(files[0]).names
has_sync = "camera.sync_ms" in pq.read_schema(files[0]).names
has_dr = "camera.depth_rgb_ms" in pq.read_schema(files[0]).names
has_item = NR.ITEM_KEY in pq.read_schema(files[0]).names
df = pd.concat([pd.read_parquet(p, columns=cols + ([NR.STEP_KEY] if has_step else []) + (["camera.sync_ms"] if has_sync else []) + (["camera.depth_rgb_ms"] if has_dr else []) + ([NR.ITEM_KEY] if has_item else [])) for p in files], ignore_index=True)
for _, e in ep.iterrows():
    i = int(e["episode_index"]); d = df[df.episode_index == i]
    old = NR.load_episode_json(rep, i) or {}
    t = e["tasks"]; task = t[0] if hasattr(t, "__len__") and not isinstance(t, str) and len(t) else t
    s = NR.episode_stats(i, fps, np.stack(d["action"].to_numpy()), np.stack(d["observation.state"].to_numpy()), e.get("episode_success"), task,
                         np.hstack([np.ravel(v) for v in d["complementary_info.is_intervention"].to_numpy()]),
                         steps=np.hstack([np.ravel(v) for v in d[NR.STEP_KEY].to_numpy()]) if has_step else None,
                         sync_ms=np.hstack([np.ravel(v) for v in d["camera.sync_ms"].to_numpy()]).astype(float) if has_sync else None,
                         depth_rgb_ms=np.stack([np.ravel(v) for v in d["camera.depth_rgb_ms"].to_numpy()])[:, 0].astype(float) if has_dr else None)
    if old.get("phases") and old["phases"][0]["name"] != "未记录":      # 录制时记下的大阶段
        s["phases"] = old["phases"]
        for sg in s["segments"]:
            sg["phase"] = next((p["name"] for p in s["phases"] if p["t0"] <= sg["t0"] < p["t1"]), s["phases"][-1]["name"])
    s["images"] = old.get("images", {}); s["recorded"] = old.get("recorded", s["recorded"])
    if has_item: s["item_rec"] = int(np.ravel(d[NR.ITEM_KEY].to_numpy()[0])[0])
    NR.save_episode_json(rep, s)
s = NR.build_site(root)
print(f"\n共 {s['episodes']} 条：成功 {s['success']}、失败 {s['failure']}、未标注 {s['unlabeled']}，总时长 {s['total_minutes']} 分钟")
print(f"学长数据标准：满足 {s['check_ok']} / {s['check_total']} 项")
for c in s["checks"]:
    if c["status"] != "ok": print(f"  {'✗' if c['status'] == 'bad' else '!'} {c['item']}：{c['actual']}")
print(f"网页：~/nero_report_serve.sh  →  http://127.0.0.1:8766/#{root.name}")
