#!/usr/bin/env python3
"""删除数据集里的某几条（用 LeRobot 自带的 delete_episodes），并同步我们自己的附加文件：
深度 depth/、报告 report/（单条统计、截图、step 修正）、物品标签 _labels/<名字>.json。
后面的条会顺次往前补号（删掉第 3 条，原来的第 4 条变成第 3 条）。原数据集整个移到 lerobot_data/_trash/ 备份。
录制程序在运行时不要用。
用法：python ~/nero_delete.py /data/panhe/lerobot_data/<名字> 3 [5 ...]"""
import datetime, json, logging, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nero_report_lib as NR

if len(sys.argv) < 3: sys.exit(__doc__)
root = Path(sys.argv[1]).resolve(); name = root.name; data_root = root.parent
drop = sorted({int(x) for x in sys.argv[2:]})
if subprocess.run(["pgrep", "-f", "nero_record.py"], capture_output=True).returncode == 0:
    sys.exit("录制程序正在运行，先按 Esc 结束录制再删除")
logging.basicConfig(level=logging.WARNING)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import delete_episodes

ds = LeRobotDataset(f"panhe/{name}", root=root)
total = ds.meta.total_episodes
bad = [i for i in drop if not 0 <= i < total]
if bad: sys.exit(f"没有第 {bad} 条（共 {total} 条，编号 0–{total - 1}）")
keep = [i for i in range(total) if i not in drop]
m = {old: new for new, old in enumerate(keep)}            # 旧编号 → 新编号
print(f"共 {total} 条，删除第 {drop} 条，剩 {len(keep)} 条。后面的条会往前补号：" + "，".join(f"{o}→{n}" for o, n in m.items() if o != n))
if input("确认删除？输入 yes：").strip() != "yes": sys.exit("已取消")

tmp = data_root / f"{name}.tmp_delete"
shutil.rmtree(tmp, ignore_errors=True)
delete_episodes(ds, drop, output_dir=tmp, repo_id=f"panhe/{name}")

# ---- 深度 ----
if (root / "depth").exists():
    for camdir in (root / "depth").iterdir():
        if not camdir.is_dir() or camdir.name.startswith("."): continue
        (tmp / "depth" / camdir.name).mkdir(parents=True, exist_ok=True)
        for old, new in m.items():
            src = camdir / f"episode_{old:06d}.mkv"
            if src.exists(): shutil.copy2(src, tmp / "depth" / camdir.name / f"episode_{new:06d}.mkv")
    di = root / "depth" / "depth_info.json"
    if di.exists():
        info = json.loads(di.read_text()); eps = info.get("episodes", {})
        info["episodes"] = {str(m[int(k)]): v for k, v in eps.items() if int(k) in m}
        (tmp / "depth" / "depth_info.json").write_text(json.dumps(info, indent=1))

# ---- 报告：单条统计、截图、step 修正 ----
rep, nrep = root / "report", tmp / "report"
(nrep / "episodes").mkdir(parents=True, exist_ok=True); (nrep / "img").mkdir(exist_ok=True)
for old, new in m.items():
    e = NR.load_episode_json(rep, old)
    if not e: continue
    e["episode"] = new
    for cam, lst in (e.get("images") or {}).items():
        for x in lst:
            src = rep / x["file"]; fn = Path(x["file"]).name.replace(f"ep{old:04d}_", f"ep{new:04d}_", 1)
            if src.exists(): shutil.copy2(src, nrep / "img" / fn)
            x["file"] = f"img/{fn}"
    NR.save_episode_json(nrep, e)
ed = NR.load_edits(rep)
if ed: NR.save_edits(nrep, {str(m[int(k)]): v for k, v in ed.items() if int(k) in m})
if (root / "meta" / "nero_labels.json").exists(): shutil.copy2(root / "meta" / "nero_labels.json", tmp / "meta" / "nero_labels.json")

# ---- 物品标签 ----
lab_p = NR.labels_path(data_root, name)
if lab_p.exists():
    lab = NR.load_labels(data_root, name)
    for key in ("episodes", "applied"):
        lab[key] = {str(m[int(k)]): v for k, v in lab[key].items() if int(k) in m}
    shutil.copy2(lab_p, lab_p.with_suffix(f".json.bak_{datetime.datetime.now():%H%M%S}"))
    NR.save_labels(data_root, name, lab)

# ---- 换上新目录，旧的进回收站 ----
trash = data_root / "_trash"; trash.mkdir(exist_ok=True)
bak = trash / f"{name}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
shutil.move(str(root), bak); shutil.move(str(tmp), root)
NR.build_site(root)
print(f"完成：剩 {len(keep)} 条。原数据集备份在 {bak}（确认没问题后可以删掉）")
print(f"核对：python ~/nero_verify.py {root}")
