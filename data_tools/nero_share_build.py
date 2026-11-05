#!/usr/bin/env python3
"""生成 8777 汇总页：/data/panhe/share/index.html + 指向数据集和导出包的软链接。
每个任务预设（~/nero_tasks.json）一张卡片：条数、时长、物品分布、step、标准检查、报告 / 下载 / 原始文件链接、大小和 md5。
以后加数据集、重新导出后重跑一次即可。只读，不改数据集。
用法：python3 ~/nero_share_build.py [lerobot_data 目录] [export 目录] [share 目录]"""
import datetime, html, json, os, sys
from pathlib import Path

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "/data/panhe/lerobot_data").resolve()
EXP = Path(sys.argv[2] if len(sys.argv) > 2 else "/data/panhe/export").resolve()
SHARE = Path(sys.argv[3] if len(sys.argv) > 3 else "/data/panhe/share")
TASKS = Path(__file__).with_name("nero_tasks.json")
E = lambda s: html.escape(str(s if s is not None else ""))


def size(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB": return f"{n:.1f} {u}" if u != "B" else f"{n} B"
        n /= 1024


def link(name, target):
    p = SHARE / name
    if p.is_symlink() or p.exists():
        if p.is_symlink() and os.readlink(p) == str(target): return
        p.unlink()
    p.symlink_to(target)


tasks = json.loads(TASKS.read_text(encoding="utf-8"))
md5 = {}
if (EXP / "md5.txt").exists():
    for ln in (EXP / "md5.txt").read_text().splitlines():
        if ln.strip(): h, f = ln.split(None, 1); md5[f.strip().lstrip("*")] = h
SHARE.mkdir(parents=True, exist_ok=True)
cards, tot_eps, tot_min = [], 0, 0.0
for key, t in sorted(tasks.items(), key=lambda kv: kv[1].get("id", 99)):
    name = t.get("dataset"); root = DATA / name
    sp = root / "report" / "summary.json"
    if not t.get("task") or not sp.exists(): continue
    s = json.loads(sp.read_text(encoding="utf-8"))
    eps, items = s.get("items", []), (s.get("labels") or {}).get("items", {})
    il = (s.get("labels") or {}).get("item_label") or t.get("item_label", "物品")
    n = s.get("episodes", len(eps)); tot_eps += n; tot_min += float(s.get("total_minutes") or 0)
    avg = sum(e.get("length_s", 0) for e in eps) / max(1, len(eps))
    cnt = {k: sum(1 for e in eps if str(e.get("item")) == k) for k in items}
    fs = s.get("fail") if t.get("failure") else None
    okn, okl = (s.get("failure", n), "失败") if fs else (s.get("success", n), "成功")
    fdist = ("".join(f'<div class="it"><span class="in">F{E(k)} · {E(v)}</span><span class="ib"><i style="width:{fs["count"].get(k, 0) / max(1, max(fs["count"].values())) * 100:.0f}%"></i></span><span class="ic">{fs["count"].get(k, 0)}</span></div>' for k, v in fs["types"].items())
             if fs else "")
    mx = max([1] + list(cnt.values()))
    bad = sum(1 for e in eps if e.get("issues"))
    link(name, root)
    tgz = EXP / f"{name}.tgz"
    if tgz.exists(): link(f"{name}.tgz", tgz)
    dist = "".join(f'<div class="it"><span class="in">{E(k)} · {E(v or "未命名")}</span><span class="ib"><i style="width:{cnt[k] / mx * 100:.0f}%"></i></span><span class="ic">{cnt[k]}</span></div>' for k, v in items.items())
    steps = " → ".join(f"<b>{E(k)}</b> {E(v)}" for k, v in (s.get("step_names") or t.get("steps") or {}).items())
    dl = f'<a class="btn pri" href="{E(name)}.tgz" download>⬇ 下载 {E(name)}.tgz <small>{size(tgz.stat().st_size)}</small></a>' if tgz.exists() else '<span class="btn off">还没导出</span>'
    m5 = f'<div class="md5">md5 <code>{E(md5.get(tgz.name, "—"))}</code></div>' if tgz.exists() else ""
    cards.append(f'''<article class="card">
  <div class="ch"><span class="no">{t.get("id", "")}</span><div><h2>{E(t.get("name") or key)}</h2><div class="ds"><code>{E(name)}</code> · {E(s.get("task", t.get("task")))}</div></div></div>
  <div class="kp"><div><b>{n}</b><span>条</span></div><div><b>{okn}</b><span>{okl}</span></div><div><b>{float(s.get("total_minutes") or 0):.1f}</b><span>分钟</span></div>
    <div><b>{avg:.1f}</b><span>秒 / 条</span></div><div><b class="{'ok' if s.get('check_ok') == s.get('check_total') else 'warn'}">{s.get("check_ok", "—")}/{s.get("check_total", "—")}</b><span>数据标准</span></div>
    <div><b class="{'ok' if not bad else 'warn'}">{bad}</b><span>问题条</span></div></div>
  <div class="st"><span class="lb">step</span>{steps}</div>
  {f'<details class="dist" open><summary>失败类型分布（停在 step k = Fk）</summary><div class="its">{fdist}</div></details>' if fs else ""}
  <details class="dist" {"open" if len(items) <= 6 else ""}><summary>{E(il)}分布（{len(items)} 个）</summary><div class="its">{dist}</div></details>
  <div class="ac">{dl}<a class="btn" href="{E(name)}/report/">📊 查看报告</a><a class="btn" href="{E(name)}/">📁 原始文件</a>{m5}</div>
</article>''')

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NERO 采集数据</title>
<style>
:root{{--bg:#f5f6f8;--card:#fff;--ink:#1b2230;--ink2:#5d6677;--line:#dfe3ea;--acc:#2f62c9;--ok:#1d8a4e;--warn:#b7701b;--sunk:#eef1f5}}
@media (prefers-color-scheme:dark){{:root{{--bg:#11151c;--card:#181e27;--ink:#e6eaf0;--ink2:#9aa4b3;--line:#2a3240;--acc:#7aa2f7;--ok:#5fcf8e;--warn:#e3a55a;--sunk:#202733}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}}
.wrap{{max-width:980px;margin:0 auto;padding:36px 16px 60px}}
header h1{{font-size:26px;margin:0 0 6px}}header p{{color:var(--ink2);margin:0;max-width:70ch}}
.tot{{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0 26px}}.tot div{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px 16px}}
.tot b{{font-size:22px;font-variant-numeric:tabular-nums;margin-right:6px}}.tot span{{color:var(--ink2);font-size:13px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px 20px;margin-bottom:18px}}
.ch{{display:flex;gap:14px;align-items:flex-start}}.no{{flex:none;width:34px;height:34px;border-radius:10px;background:var(--acc);color:#fff;display:grid;place-items:center;font-weight:800}}
.ch h2{{margin:0;font-size:19px}}.ds{{color:var(--ink2);font-size:13px;overflow-wrap:anywhere}}code{{font:12.5px ui-monospace,Menlo,Consolas,monospace}}
.kp{{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin:16px 0 12px}}.kp div{{background:var(--sunk);border-radius:10px;padding:8px 10px}}
.kp b{{display:block;font-size:20px;font-variant-numeric:tabular-nums}}.kp span{{color:var(--ink2);font-size:12px}}.ok{{color:var(--ok)}}.warn{{color:var(--warn)}}
.st{{font-size:13.5px;color:var(--ink2)}}.st b{{color:var(--ink)}}.lb{{display:inline-block;font-size:11px;border:1px solid var(--line);border-radius:6px;padding:0 6px;margin-right:8px;color:var(--ink2)}}
.dist summary{{cursor:pointer;font-size:13.5px;color:var(--ink2);margin-top:10px}}.its{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:4px 20px;margin-top:8px}}
.it{{display:grid;grid-template-columns:minmax(0,1fr) 90px 28px;gap:8px;align-items:center;font-size:13px}}.in{{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.ib{{height:6px;border-radius:9px;background:var(--sunk);overflow:hidden}}.ib i{{display:block;height:100%;background:var(--acc)}}.ic{{text-align:right;font-variant-numeric:tabular-nums}}
.ac{{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:14px}}
.btn{{display:inline-flex;gap:6px;align-items:baseline;padding:8px 14px;border-radius:10px;border:1px solid var(--line);color:var(--ink);text-decoration:none;font-size:14px}}
.btn:hover{{border-color:var(--acc)}}.btn.pri{{background:#2f62c9;border-color:#2f62c9;color:#fff}}.btn.pri small{{opacity:.85}}.btn.off{{color:var(--ink2)}}
.md5{{flex-basis:100%;font-size:12px;color:var(--ink2);overflow-wrap:anywhere}}
.how{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px 20px;font-size:14px}}.how h3{{margin:0 0 8px;font-size:15px}}
.how pre{{background:var(--sunk);border-radius:10px;padding:10px 12px;overflow:auto;font:12.5px/1.5 ui-monospace,Menlo,Consolas,monospace}}
footer{{color:var(--ink2);font-size:12.5px;margin-top:20px}}
@media (max-width:720px){{.kp{{grid-template-columns:repeat(3,1fr)}}.its{{grid-template-columns:1fr}}}}
</style></head><body><div class="wrap">
<header><h1>NERO 机械臂采集数据</h1><p>人工遥操作示范数据：成功示范，以及标注了失败类型的失败示范（数据集名带 _fail），10 Hz，LeRobot v3 格式；头部 + 腕部 RGB 视频，深度对齐到彩色、单独无损存储在 <code>depth/</code>；每帧带 step 编号、物品编号、主臂夹爪和两相机时间戳。</p></header>
<div class="tot"><div><b>{len(cards)}</b><span>个任务</span></div><div><b>{tot_eps}</b><span>条</span></div><div><b>{tot_min:.0f}</b><span>分钟</span></div></div>
{"".join(cards)}
<section class="how"><h3>怎么用</h3>
<p>下载 .tgz 解压，得到 LeRobot v3 数据集目录（含 <code>report/</code> 只读报告）。读取：</p>
<pre>from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("panhe/nero_weigh_v1", root="解压目录/nero_weigh_v1")
# 深度（uint16 毫米，0 = 无效）：depth/&lt;相机&gt;/episode_XXXXXX.mkv，FFV1 无损，帧号与数据集一一对应</pre>
<p>离线看报告：在解压目录运行 <code>python -m http.server 8777</code>，打开 <code>http://127.0.0.1:8777/数据集名/report/</code>。</p></section>
<footer>生成于 {now} · 数据目录 {E(DATA)} · 导出包 {E(EXP)}</footer>
</div></body></html>'''
(SHARE / "index.html").write_text(page, encoding="utf-8")
print(f"已更新 {SHARE / 'index.html'}：{len(cards)} 个任务，{tot_eps} 条")
