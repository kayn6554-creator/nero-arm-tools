#!/usr/bin/env python3
"""把网页上的修正写进数据集：step 分界（step_index）和物品标签（item_index）。
同时重算这两个字段的每条统计和 meta/stats.json，物品名字写到 meta/nero_labels.json。
录制程序在运行时不要用。原文件先备份到 report/backup_<时间>/。
用法：python ~/nero_steps_apply.py /data/panhe/lerobot_data/<名字>"""
import datetime, glob, json, shutil, subprocess, sys
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nero_report_lib as NR
from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats

root = Path(sys.argv[1]).resolve(); rep = root / "report"
NR.use_task(root.name)        # step 名字按任务预设
if subprocess.run(["pgrep", "-f", "nero_record.py"], capture_output=True).returncode == 0:
    sys.exit("录制程序正在运行，先按 Esc 结束录制再写入")
info = json.loads((root / "meta/info.json").read_text()); fps = info["fps"]; F = info["features"]
edits = {int(k): v for k, v in NR.load_edits(rep).items() if not v.get("applied")} if NR.STEP_KEY in F else {}
lab = NR.load_labels(root.parent, root.name)
item_new = {int(k): int(v) for k, v in lab["episodes"].items() if lab["applied"].get(k) != v} if NR.ITEM_KEY in F else {}
if not edits and not item_new:
    (root / "meta/nero_labels.json").write_text(json.dumps(dict(items=lab["items"], task_key=lab.get("task_key")), ensure_ascii=False, indent=1), encoding="utf-8")
    sys.exit("没有待写入的修正（物品名字已同步到 meta/nero_labels.json）")

bak = rep / f"backup_{datetime.datetime.now():%Y%m%d_%H%M%S}"
files = sorted(glob.glob(str(root / "data/chunk-*/*.parquet"))); mfiles = sorted(glob.glob(str(root / "meta/episodes/chunk-*/*.parquet")))
for f in files + mfiles + [str(root / "meta/stats.json")]:
    dst = bak / Path(f).relative_to(root); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, dst)
print(f"已备份到 {bak}")

def replace_col(tbl, name, values):
    i = tbl.schema.get_field_index(name); return tbl.set_column(i, tbl.schema.field(i), pa.array(values, type=tbl.schema.field(i).type))
def as_col(t, K, arr):
    ty = t.schema.field(K).type
    return [[int(v)] for v in arr] if (pa.types.is_list(ty) or pa.types.is_fixed_size_list(ty)) else [int(v) for v in arr]

jobs = []   # (字段, 要改的条, 计算新值的函数)
if edits: jobs.append((NR.STEP_KEY, set(edits), lambda ep, fi: NR.step_of_time(fi / fps, edits[ep]["bounds"])))
if item_new: jobs.append((NR.ITEM_KEY, set(item_new), lambda ep, fi: np.full(len(fi), item_new[ep], dtype=np.int64)))

# 先全部算好，再一起写，避免写到一半出错
tables = {f: pq.read_table(f) for f in files}; touched = set(); changed = {}; ep_vals = {}
for K, eps_set, fn in jobs:
    ep_vals[K] = {}
    for f, t in tables.items():
        epi = t.column("episode_index").to_numpy(); fi = t.column("frame_index").to_numpy()
        cur = np.array([np.ravel(v)[0] for v in t.column(K).to_pylist()], dtype=np.int64); new = cur.copy()
        for ep in eps_set:
            m = epi == ep
            if m.any(): new[m] = fn(ep, fi[m])
        for ep in np.unique(epi): ep_vals[K][int(ep)] = new[epi == ep]
        if (new != cur).any():
            for ep in np.unique(epi[new != cur]): changed[(K, int(ep))] = int((new[epi == ep] != cur[epi == ep]).sum())
            tables[f] = replace_col(t, K, as_col(t, K, new)); touched.add(f)
mt = {f: pq.read_table(f) for f in mfiles}; mtouched = set()
sp = root / "meta/stats.json"; stats = json.loads(sp.read_text())
for K, eps_set, _ in jobs:
    ep_stats = {ep: get_feature_stats(v.reshape(-1, 1).astype(np.float64), axis=0, keepdims=False) for ep, v in ep_vals[K].items()}
    for f, t in mt.items():
        eps = t.column("episode_index").to_pylist(); cols = [c for c in t.column_names if c.startswith(f"stats/{K}/")]
        for col in cols:
            stat = col.split("/")[-1]; vals = t.column(col).to_pylist()
            for r, ep in enumerate(eps):
                if ep in eps_set and stat in ep_stats[ep]: vals[r] = np.asarray(ep_stats[ep][stat]).tolist()
            t = replace_col(t, col, vals)
        if cols: mt[f] = t; mtouched.add(f)
    agg = aggregate_stats([{K: ep_stats[ep]} for ep in sorted(ep_stats)])[K]
    stats[K] = {k: np.asarray(v).tolist() for k, v in agg.items()}

for f in touched: pq.write_table(tables[f], f + ".tmp"); Path(f + ".tmp").replace(f)
for f in mtouched: pq.write_table(mt[f], f + ".tmp"); Path(f + ".tmp").replace(f)
sp.write_text(json.dumps(stats, indent=4))

now = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}"
if edits:
    all_edits = NR.load_edits(rep)
    for ep in edits: all_edits[str(ep)]["applied"] = True; all_edits[str(ep)]["applied_at"] = now
    NR.save_edits(rep, all_edits)
    for ep in edits:
        e = NR.load_episode_json(rep, ep); e["steps"] = NR.steps_from_bounds(edits[ep]["bounds"], e["length_s"]); NR.save_episode_json(rep, e)
if item_new:
    lab = NR.load_labels(root.parent, root.name)
    for ep, it in item_new.items():
        lab["applied"][str(ep)] = it
        e = NR.load_episode_json(rep, ep); e["item_rec"] = it; NR.save_episode_json(rep, e)
    NR.save_labels(root.parent, root.name, lab)
(root / "meta/nero_labels.json").write_text(json.dumps(dict(items=NR.load_labels(root.parent, root.name)["items"], task_key=lab.get("task_key"),
                                                             item_field=NR.ITEM_KEY, step_names=NR.STEP_NAMES), ensure_ascii=False, indent=1), encoding="utf-8")
NR.build_site(root)
name = {NR.STEP_KEY: "step", NR.ITEM_KEY: "物品"}
print("已写入：" + "，".join(f"第 {ep} 条 {name[K]} 改了 {n} 帧" for (K, ep), n in sorted(changed.items(), key=lambda x: (x[0][1], x[0][0]))) if changed else "数据本来就和修正一致")
