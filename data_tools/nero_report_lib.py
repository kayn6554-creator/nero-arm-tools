"""NERO 采集报告 v2：单条统计（大阶段、动作分段、夹住检测、截图）+ 数据标准检查 + 总览网站。
网站是一个单页应用：lerobot_data/index.html 读取 datasets.json 和各数据集 report/ 下的 JSON，打开就是数据总览。"""
import csv, datetime, json
from pathlib import Path
import numpy as np

JN = ["j1 底座转", "j2 大臂", "j3 大臂转", "j4 小臂", "j5 小臂转", "j6 腕摆", "j7 腕转"]
STATE_NAMES = [f"joint{i}.pos" for i in range(1, 8)] + ["gripper.pos"]
APP = Path(__file__).with_name("nero_report_app.html")
ITEM_KEY = "item_index"          # 每帧的物品编号（0 = 未标）；名字在 _labels/<数据集>.json
TASKS_FILE = Path(__file__).with_name("nero_tasks.json")
STEP_KEY = "step_index"
STEP_NAMES = {1: "抓", 2: "移动", 3: "放置", 4: "离开"}
STEP_NAMES_STR = "".join(str(k) for k in STEP_NAMES)
DEFAULT_STEPS = dict(STEP_NAMES)
HOLD_GAP_MM, HOLD_MIN_S = 8.0, 1.0   # 主臂指令比从臂实际更合拢 8 mm 以上并持续 1 s = 夹住物体


def _label(v):
    if isinstance(v, (bool, np.bool_)): return "success" if v else "failure"
    v = str(v).lower() if v is not None else ""
    return v if v in ("success", "failure") else "unlabeled"


def _runs(mask):
    out, s = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and s is None: s = i
        if not m and s is not None: out.append((s, i - 1)); s = None
    return out


def phases_from_flags(flags, n, fps):
    if flags is None or len(flags) != n: return [dict(name="未记录", t0=0.0, t1=round(n / fps, 2))]
    r = _runs(np.asarray(flags, bool))
    if not r: return [dict(name="任务遥操作", t0=0.0, t1=round(n / fps, 2))]
    a, b = r[0][0], r[-1][1] + 1; ph = []
    if a > 0: ph.append(dict(name="复位", t0=0.0, t1=round(a / fps, 2)))
    ph.append(dict(name="定位移动", t0=round(a / fps, 2), t1=round(b / fps, 2)))
    if b < n: ph.append(dict(name="任务遥操作", t0=round(b / fps, 2), t1=round(n / fps, 2)))
    return ph


def episode_stats(idx, fps, act, st, label, task, inter=None, pos_flags=None, steps=None, sync_ms=None, depth_rgb_ms=None):
    act, st = np.asarray(act, float), np.asarray(st, float)
    n = len(st); t = np.arange(n) / fps
    q = st[:, :7]; g = st[:, 7] * 1000 if st.shape[1] > 7 else np.zeros(n)
    gc = act[:, 7] * 1000 if act.shape[1] > 7 else g
    k = max(1, round(0.1 * fps)) if n > 3 else 0      # 从臂固有延迟约 0.1 s（30 Hz 为 3 帧，10 Hz 为 1 帧）
    diff = float(np.abs(act[:n - k, :7] - st[k:, :7]).mean() * 180 / np.pi) if n else 0.0
    v = np.r_[0, np.linalg.norm(np.diff(q, axis=0), axis=1)] * fps
    v = np.convolve(v, np.ones(5) / 5, mode="same")
    segs = []
    for a, b in _runs(v > np.radians(3)):
        if (b - a + 1) / fps < 0.3: continue
        d = np.degrees(q[b] - q[a]); top = np.argsort(-np.abs(d))[:2]
        desc = "，".join(f"{JN[j]} {d[j]:+.0f}°" for j in top if abs(d[j]) >= 2) or "小幅调整"
        segs.append(dict(kind="move", t0=round(t[a], 2), t1=round(t[b], 2), text=f"移动（{desc}）"))
    w = max(1, int(1.5 * fps)); i = 0
    while i + 1 < n:
        j = min(n - 1, i + w); seg = g[i:j + 1]; lo, hi = int(np.argmin(seg)) + i, int(np.argmax(seg)) + i
        if g[i] - g[lo] > 10 and lo > i:
            segs.append(dict(kind="close", t0=round(t[i], 2), t1=round(t[lo], 2), text=f"夹爪闭合 {g[i]:.0f}→{g[lo]:.0f} mm")); i = lo; continue
        if g[hi] - g[i] > 10 and hi > i:
            segs.append(dict(kind="open", t0=round(t[i], 2), t1=round(t[hi], 2), text=f"夹爪张开 {g[i]:.0f}→{g[hi]:.0f} mm")); i = hi; continue
        i += max(1, w // 3)
    # 夹住检测：从臂夹爪没法直接告诉我们“夹紧了没有”，但主臂指令和从臂实际宽度之差可以
    holds = []
    for a, b in _runs((gc < g - HOLD_GAP_MM) & (g > 3)):
        if (b - a + 1) / fps < HOLD_MIN_S: continue
        wmm = float(np.median(g[a:b + 1])); cmd = float(np.median(gc[a:b + 1]))
        holds.append(dict(t0=round(t[a], 2), t1=round(t[b], 2), width_mm=round(wmm, 1), cmd_mm=round(cmd, 1)))
        segs.append(dict(kind="hold", t0=round(t[a], 2), t1=round(t[b], 2), text=f"夹住物体 {t[b]-t[a]:.1f} s（实际宽 {wmm:.0f} mm，主臂指令 {cmd:.0f} mm）"))
    segs.sort(key=lambda s: s["t0"])
    phases = phases_from_flags(pos_flags, n, fps)
    for s in segs:
        s["phase"] = next((p["name"] for p in phases if p["t0"] <= s["t0"] < p["t1"]), phases[-1]["name"])
    step_segs = None
    if steps is not None and len(steps) == n:
        steps = np.asarray(steps).astype(int).ravel(); step_segs = []
        for v in [int(x) for x in np.unique(steps)]:
            for a, b in _runs(steps == v):
                step_segs.append(dict(step=v, name=STEP_NAMES.get(v, f"step {v}"), t0=round(t[a], 2), t1=round((b + 1) / fps, 2)))
        step_segs.sort(key=lambda x: x["t0"])
        for s in segs:
            s["step"] = next((x["name"] for x in step_segs if x["t0"] <= s["t0"] < x["t1"]), "")
    cnt = lambda kd: sum(s["kind"] == kd for s in segs)
    step = max(1, n // 300)
    out = dict(episode=int(idx), fps=fps, frames=n, length_s=round(n / fps, 2), result=_label(label), task=str(task),
               action_state_diff_deg=round(diff, 3), gripper_range_mm=round(float(g.max() - g.min()), 1) if n else 0.0,
               intervention_ratio=round(float(np.mean(inter)) if inter is not None and len(inter) else 0.0, 3),
               motion_summary=f"移动 {cnt('move')} 段 · 闭合 {cnt('close')} · 张开 {cnt('open')} · 夹住 {len(holds)}",
               counts=dict(move=cnt("move"), close=cnt("close"), open=cnt("open"), hold=len(holds)),
               phases=phases, segments=segs, holds=holds, steps=step_segs, sync=None,
               trace=dict(t=[round(x, 2) for x in t[::step]], joints=np.round(np.degrees(q[::step]), 1).T.tolist(),
                          gripper=np.round(g[::step], 1).tolist(), gripper_cmd=np.round(gc[::step], 1).tolist()),
               images={}, recorded=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if depth_rgb_ms is not None and len(depth_rgb_ms) and np.isfinite(np.asarray(depth_rgb_ms, float)).any():
        s = np.abs(np.asarray(depth_rgb_ms, float)); s = s[np.isfinite(s)]
        out["depth_sync"] = dict(mean_ms=round(float(s.mean()), 2), max_ms=round(float(s.max()), 2), over_20ms=int((s > 20).sum()))
    if sync_ms is not None and len(sync_ms) and np.isfinite(np.asarray(sync_ms, float)).any():
        s = np.abs(np.asarray(sync_ms, float)); s = s[np.isfinite(s)]
        out["sync"] = dict(mean_ms=round(float(s.mean()), 1), p95_ms=round(float(np.percentile(s, 95)), 1), max_ms=round(float(s.max()), 1),
                           over_20ms=int((s > 20).sum()))
    return out


def save_episode_json(report_dir, stats):
    d = Path(report_dir) / "episodes"; d.mkdir(parents=True, exist_ok=True)
    (d / f"ep_{stats['episode']:04d}.json").write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")


def load_episode_json(report_dir, idx):
    p = Path(report_dir) / "episodes" / f"ep_{idx:04d}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def capture_live_episode(ds, extra_meta, pos_flags=None):
    from PIL import Image
    b = ds.episode_buffer; report = Path(ds.root) / "report"
    idx = int(np.ravel(b["episode_index"])[0]); n = int(b["size"])
    act, st = np.stack(b["action"]), np.stack(b["observation.state"])
    iv = b.get("complementary_info.is_intervention")
    inter = np.hstack([np.ravel(x) for x in iv]) if iv else None
    sv = b.get(STEP_KEY); sy = b.get("camera.sync_ms"); dr = b.get("camera.depth_rgb_ms")
    drf = np.stack([np.ravel(x) for x in dr])[:, 0].astype(float) if dr else None   # 第 0 列 = front（头部）
    steps = np.hstack([np.ravel(x) for x in sv]) if sv else None
    sync = np.hstack([np.ravel(x) for x in sy]).astype(float) if sy else None
    stats = episode_stats(idx, ds.fps, act, st, (extra_meta or {}).get("episode_success"), b["task"][0] if b.get("task") else "", inter, pos_flags, steps, sync, drf)
    iv2 = b.get(ITEM_KEY)
    stats["item_rec"] = int(np.ravel(iv2[0])[0]) if iv2 else None
    imgdir = report / "img"; imgdir.mkdir(parents=True, exist_ok=True)
    for key in [k for k in b if k.startswith("observation.images.")]:
        cam = key.split(".")[-1]; frames = b[key]; stats["images"][cam] = []
        for tag, fi in (("开始", 0), ("中间", n // 2), ("结束", n - 1)):
            src = frames[fi]
            img = Image.open(src) if isinstance(src, (str, Path)) else Image.fromarray(np.asarray(src))
            name = f"ep{idx:04d}_{cam}_{fi}.jpg"
            img.convert("RGB").resize((320, 240)).save(imgdir / name, quality=80)
            stats["images"][cam].append(dict(tag=tag, t=round(fi / ds.fps, 1), file=f"img/{name}"))
    save_episode_json(report, stats)
    return stats


# ---------------------------------------------------------------- step 分界修正（网页里拖动，存在 report/step_edits.json）
def load_edits(rep):
    p = Path(rep) / "step_edits.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        return {}


def save_edits(rep, edits):
    p = Path(rep) / "step_edits.json"; tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(edits, ensure_ascii=False, indent=1), encoding="utf-8"); tmp.replace(p)


def steps_from_bounds(bounds, length):
    edges = [0.0] + [float(b) for b in bounds] + [float(length)]
    return [dict(step=k + 1, name=STEP_NAMES.get(k + 1, f"step {k + 1}"), t0=round(edges[k], 2), t1=round(edges[k + 1], 2)) for k in range(len(edges) - 1)]


def step_of_time(t, bounds):
    t = np.asarray(t, float)
    return 1 + sum((t >= float(b)).astype(int) for b in bounds)


# ---------------------------------------------------------------- 物品标签（网页输入，存在 lerobot_data/_labels/<数据集>.json）
def fail_cfg(t):
    """失败数据集的预设：{per_type, types: {"1": 名字}, definitions}；不是失败数据集返回 None。"""
    return (t or {}).get("failure") or None


def load_tasks():
    try:
        return json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def task_for_dataset(name):
    for k, t in load_tasks().items():
        if t.get("dataset") == name or name.startswith(t.get("dataset", "\0").rsplit("_v", 1)[0] + "_v"):
            return k, t
    return None, None


def use_task(name):
    """按数据集对应的任务预设切换 step 名字（就地改 STEP_NAMES，所有引用都跟着变）。没有预设就用默认 4 步。"""
    global STEP_NAMES_STR
    _, t = task_for_dataset(name)
    st = {int(k): v for k, v in ((t or {}).get("steps") or {}).items()} or DEFAULT_STEPS
    STEP_NAMES.clear(); STEP_NAMES.update(dict(sorted(st.items()))); STEP_NAMES_STR = "".join(str(k) for k in STEP_NAMES)
    return t


def labels_path(data_root, name):
    return Path(data_root) / "_labels" / f"{name}.json"


def load_labels(data_root, name):
    p = labels_path(data_root, name)
    try:
        lab = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        lab = {}
    k, t = task_for_dataset(name)
    if "items" not in lab and (t or {}).get("items_from"):      # 失败数据集：物品名字沿用对应的成功数据集
        try:
            lab["items"] = json.loads(labels_path(data_root, t["items_from"]).read_text(encoding="utf-8")).get("items", {})
        except Exception:
            pass
    lab.setdefault("task_key", k)
    lab.setdefault("items", {str(i): "" for i in range(1, (t or {}).get("item_slots", 0) + 1)})
    lab.setdefault("per_item", (t or {}).get("per_item", 10))
    lab.setdefault("current", 1 if lab["items"] else 0)
    lab.setdefault("episodes", {})            # 网页上改过的：{条: 物品号}
    lab.setdefault("applied", {})             # 已写进数据集的：{条: 物品号}
    return lab


def save_labels(data_root, name, lab):
    p = labels_path(data_root, name); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp"); tmp.write_text(json.dumps(lab, ensure_ascii=False, indent=1), encoding="utf-8"); tmp.replace(p)


def item_name(lab, i):
    i = int(i or 0)
    if i <= 0: return "未标"
    nm = (lab.get("items") or {}).get(str(i), "")
    return nm or f"物品 {i}"


# ---------------------------------------------------------------- 数据标准检查
def _fail_steps_ok(st):
    return bool(st) and st == list(range(1, len(st) + 1))


def _find(f, *words):
    return [k for k in f if all(w in k.lower() for w in words)]


def standard_check(root, eps):
    """返回 [{group,item,require,actual,status,why}]。group A = 学长数据标准（10-28），B = Evo-RL 训练兼容。"""
    rows = []
    try:
        info = json.loads((Path(root) / "meta/info.json").read_text())
    except Exception as e:
        return [dict(group="A", item="meta/info.json", require="存在且可读", actual=f"读取失败：{e}", status="bad", why="")]
    f = info.get("features", {}); fps = info.get("fps")
    def add(group, item, req, got, ok, level="bad", why=""):
        rows.append(dict(group=group, item=item, require=req, actual=str(got), status="ok" if ok else level, why=why))
    cams = {"head": "front", "wrist": "wrist"}
    # A · 学长标准
    add("A", "数据格式", "LeRobot v3.0", info.get("codebase_version"), info.get("codebase_version") == "v3.0")
    add("A", "采集频率", "10 Hz", f"{fps} Hz", fps == 10, why="相机和关节都按 10 Hz 存一帧")
    for cn, key in cams.items():
        ft = f.get(f"observation.images.{key}", {})
        add("A", f"{'头部' if cn == 'head' else '腕部'} RGB", "有，640×480", f"{ft.get('dtype')} {ft.get('shape')}" if ft else "缺失",
            bool(ft) and list(ft.get("shape", []))[:2] == [480, 640])
    dinfo = {}
    try:
        dinfo = json.loads((Path(root) / "depth" / "depth_info.json").read_text())
    except Exception:
        pass
    for cn, key in cams.items():
        nm = f"{'头部' if cn == 'head' else '腕部'} 深度"
        if dinfo:
            per = {int(k): v.get(key) for k, v in dinfo.get("episodes", {}).items()}
            miss = [e["episode"] for e in eps if not (per.get(e["episode"]) or {}).get("ok")]
            add("A", nm, "有，16 位毫米，无损，帧号对齐", f"depth/{key}/ 无损 FFV1；" + ("每条都齐" if not miss else f"第 {miss} 条缺或帧数不符"), not miss,
                why="单独存放（不进训练），读取：nero_depth.load_depth(数据集, 条, 相机)")
        else:
            dk = _find(f, "depth", key) or _find(f, "depth", cn)
            add("A", nm, "有，16 位毫米", dk[0] if dk else "未录制", bool(dk))
    syn = [e.get("sync") for e in eps if e.get("sync")]
    if _find(f, "sync") or [k for k in f if "stamp" in k.lower() and k != "timestamp"] or syn:
        worst = max((s["max_ms"] for s in syn), default=None)
        add("A", "两相机时间差", "≤ 20 ms", f"最大 {worst} ms" if worst is not None else "有字段，未统计", worst is not None and worst <= 20)
    else:
        add("A", "两相机时间差", "≤ 20 ms", "未记录相机时间戳，无法检查", False, why="需要每帧存两个相机的硬件时间戳")
    if "camera.depth_rgb_ms" in f:
        ds_ = [e.get("depth_sync") for e in eps if e.get("depth_sync")]
        worst = max((x["max_ms"] for x in ds_), default=None)
        add("A", "头部 深度−RGB 时间差", "≤ 20 ms", f"最大 {worst} ms" if worst is not None else "有字段，未统计", worst is not None and worst <= 20)
    else:
        add("A", "头部 深度−RGB 时间差", "≤ 20 ms", "这个数据集录制时还没记录（当时按 ≤ 3 ms 配对，但没存时间戳）", False, "warn",
            why="10-28 14:30 之后新建的数据集才有 camera.depth_rgb_ms")
    ft = f.get("observation.state", {}); names = ft.get("names") if isinstance(ft.get("names"), list) else None
    add("A", "从臂关节位置", "joint1–7（rad）", "observation.state " + (", ".join(names[:7]) if names else str(ft.get("shape"))) if ft else "缺失",
        bool(ft) and (names is None or names[:7] == STATE_NAMES[:7]))
    lg = _find(f, "leader", "grip")
    add("A", "主臂夹爪", "单独记录", lg[0] if lg else "只在 action.gripper.pos 里（已限幅、换算到从臂行程）", bool(lg), "warn",
        why="从臂无法直接确认是否夹紧；报告用“主臂指令 vs 从臂实际”的差判断夹住")
    sk = _find(f, "step") or _find(f, "subtask")
    fail = fail_cfg(task_for_dataset(Path(root).name)[1])
    if sk and eps and fail:
        badst = [e["episode"] for e in eps if not _fail_steps_ok([x["step"] for x in (e.get("steps") or [])])]
        add("A", "任务分 step", "失败数据：从 1 开始按顺序，到失败的那一步结束", f"{sk[0]}；" + ("每条都正确" if not badst else f"第 {badst} 条顺序不对或没按"), not badst, "warn")
    elif sk and eps:
        want = list(STEP_NAMES)
        badst = [e["episode"] for e in eps if [x["step"] for x in (e.get("steps") or [])] != want]
        add("A", "任务分 step", "每条按顺序 1 抓 → 2 移动 → 3 放置 → 4 离开", f"{sk[0]}；" + ("每条都完整" if not badst else f"第 {badst} 条不完整或顺序不对"), not badst, "warn")
    else:
        add("A", "任务分 step", "抓 / 移动 / 放置 / 离开，按键切换", sk[0] if sk else "未记录", bool(sk))
    nu = sum(e["result"] == "unlabeled" for e in eps)
    add("A", "成败标签", "每条 s / f", f"未标注 {nu} 条", nu == 0)
    if ITEM_KEY in f:
        miss = [e["episode"] for e in eps if not e.get("item")]
        pend = [e["episode"] for e in eps if e.get("item_pending")]
        add("A", "物品标签", "每条都标了是哪个物品", ("全部已标" if not miss else f"第 {miss} 条未标") + (f"；第 {pend} 条改过还没写入数据集" if pend else ""),
            not miss, why="网页上选“当前物品”后录制会自动带上；录完也能在每条详情里改")
    # B · Evo-RL 兼容
    for key in ("action", "observation.state"):
        ft = f.get(key, {}); nm = ft.get("names") if isinstance(ft.get("names"), list) else None
        add("B", key, "8 维 joint1–7.pos + gripper.pos", str(ft.get("shape")) if ft else "缺失", bool(ft) and list(ft.get("shape", [])) == [8] and (nm is None or nm == STATE_NAMES))
    miss = [k for k in ("complementary_info.policy_action", "complementary_info.is_intervention", "complementary_info.state", "complementary_info.collector_policy_id") if k not in f]
    add("B", "Evo-RL 附加字段", "policy_action / is_intervention / state / collector_policy_id", "齐全" if not miss else f"缺 {miss}", not miss)
    if eps:
        tasks = {e["task"] for e in eps}
        add("B", "任务描述", "一个，纯文字，不带冒号", " / ".join(sorted(tasks))[:100], len(tasks) == 1 and not any("{" in t or ":" in t for t in tasks))
        ns = sum(e["result"] == "success" for e in eps); nf = sum(e["result"] == "failure" for e in eps)
        if fail:
            add("B", "成败标签", "失败数据集：每条都按 f", f"成功 {ns} / 失败 {nf}", ns == 0)
        else:
            add("B", "成败标签", "人工示范只录成功；失败在模型测试阶段产生", f"成功 {ns} / 失败 {nf}", True)
        bad = [e["episode"] for e in eps if e["action_state_diff_deg"] > 0.5]
        add("B", "动作与执行一致", "动作−状态差 < 0.5°", "全部正常" if not bad else f"第 {bad} 条偏大", not bad, "warn")
        extra = [e["episode"] for e in eps if any(p["name"] in ("复位", "定位移动") for p in e.get("phases", []))]
        add("B", "录制范围", "待和学长确认", "全部只含任务遥操作" if not extra else f"第 {extra} 条含复位 / 定位移动", not extra, "warn",
            why="新标准把“离开（复位）”算作一个 step，录制范围要跟着定")
    add("B", "头部相机有效帧率", "≥ 采集频率", f"相机 15 fps（USB 2），采集 {fps} Hz" + ("" if fps is not None and fps <= 15 else "，约一半是重复帧"), fps is not None and fps <= 15, "warn")
    return rows


# ---------------------------------------------------------------- 网站
def _thumb(e):
    im = e.get("images", {}).get("front") or e.get("images", {}).get("wrist") or []
    return im[len(im) // 2]["file"] if im else None


def write_index(data_root):
    """在 lerobot_data 根目录写 datasets.json 和单页应用 index.html。"""
    data_root = Path(data_root); items = []
    for s in sorted(data_root.glob("*/report/summary.json")):
        try:
            d = json.loads(s.read_text(encoding="utf-8"))
        except Exception:
            continue
        items.append(dict(name=s.parent.parent.name, generated=d.get("generated"), episodes=d.get("episodes"), success=d.get("success"),
                          failure=d.get("failure"), total_minutes=d.get("total_minutes"), check_ok=d.get("check_ok"), check_total=d.get("check_total")))
    have = {x["name"] for x in items}
    live = None
    try:
        live = json.loads((data_root / "_live.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    extra = set(p.stem for p in (data_root / "_labels").glob("*.json")) if (data_root / "_labels").exists() else set()
    if live: extra.add(live.get("dataset"))
    extra |= {t["dataset"] for t in load_tasks().values() if t.get("task") and t.get("dataset")}   # 任务预设的数据集：还没录也先列出来，方便先填物品
    for nm in sorted(extra - have - {None}):
        items.append(dict(name=nm, generated=None, episodes=0, empty=True))
    for x in items:
        x["live"] = bool(live and x["name"] == live.get("dataset"))
        lp = labels_path(data_root, x["name"])
        if not lp.exists() and task_for_dataset(x["name"])[1]: save_labels(data_root, x["name"], load_labels(data_root, x["name"]))
        x["labels_mtime"] = lp.stat().st_mtime if lp.exists() else None
    items.sort(key=lambda x: (not x["live"], -(datetime.datetime.strptime(x["generated"], "%Y-%m-%d %H:%M:%S").timestamp() if x.get("generated") else 0)))
    (data_root / "datasets.json").write_text(json.dumps(dict(generated=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), datasets=items, tasks=load_tasks()), ensure_ascii=False, indent=1), encoding="utf-8")
    if APP.exists():
        (data_root / "index.html").write_text(APP.read_text(encoding="utf-8"), encoding="utf-8")


def build_site(root):
    root = Path(root); rep = root / "report"; rep.mkdir(exist_ok=True)
    task = use_task(root.name) or {}
    fail = fail_cfg(task)
    eps = sorted((json.loads(p.read_text(encoding="utf-8")) for p in (rep / "episodes").glob("ep_*.json")), key=lambda e: e["episode"])
    edits = load_edits(rep)
    for e in eps:
        ed = edits.get(str(e["episode"]))
        if ed:
            e["steps_orig"] = e.get("steps"); e["steps"] = steps_from_bounds(ed["bounds"], e["length_s"])
            e["steps_edit"] = dict(applied=bool(ed.get("applied")), updated=ed.get("updated"))
            for sg in e.get("segments", []):
                sg["step"] = next((x["name"] for x in e["steps"] if x["t0"] <= sg["t0"] < x["t1"]), "")
    lab = load_labels(root.parent, root.name)
    for e in eps:
        rec = e.get("item_rec")
        cur = lab["episodes"].get(str(e["episode"]), lab["applied"].get(str(e["episode"]), rec))
        e["item"] = int(cur) if cur else 0
        e["item_name"] = item_name(lab, e["item"])
        e["item_pending"] = str(e["episode"]) in lab["episodes"] and lab["episodes"][str(e["episode"])] != rec and lab["applied"].get(str(e["episode"])) != lab["episodes"][str(e["episode"])]
    has_item_f = False
    try:
        has_item_f = ITEM_KEY in json.loads((root / "meta/info.json").read_text()).get("features", {})
    except Exception:
        pass
    for e in eps:   # 每条的问题，网页上标在卡片和物品上
        iss = []
        st = [x["step"] for x in (e.get("steps") or [])]
        if fail:
            e["fail_type"] = len(st) if e["result"] == "failure" and _fail_steps_ok(st) else 0
            if e["result"] != "failure": iss.append("失败数据集里这条标成了" + ("成功" if e["result"] == "success" else "未标注"))
            elif e.get("steps") is not None and not _fail_steps_ok(st): iss.append("step " + ("→".join(map(str, st)) or "没按") + "（失败数据要从 1 开始按顺序，到失败那一步结束）")
        elif e.get("steps") is not None and st != list(STEP_NAMES): iss.append("step " + ("→".join(map(str, st)) or "没按") + ("（缺 " + "、".join(f"{k} {v}" for k, v in STEP_NAMES.items() if k not in st) + "）" if set(STEP_NAMES) - set(st) else "（顺序不对）"))
        if (e.get("sync") or {}).get("max_ms", 0) > 20: iss.append(f"两相机时间差 {e['sync']['max_ms']} ms")
        if (e.get("depth_sync") or {}).get("max_ms", 0) > 20: iss.append(f"头部深度−RGB {e['depth_sync']['max_ms']} ms")
        if has_item_f and not e.get("item"): iss.append("没标物品")
        e["issues"] = iss
    n = len(eps); ns = sum(e["result"] == "success" for e in eps); nf = sum(e["result"] == "failure" for e in eps); nu = n - ns - nf
    rate = ns / (ns + nf) if ns + nf else 0.0; tot = sum(e["length_s"] for e in eps)
    chk = standard_check(root, eps)
    A = [r for r in chk if r["group"] == "A"]
    gen = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        info = json.loads((root / "meta/info.json").read_text())
    except Exception:
        info = {}
    items = [dict(episode=e["episode"], result=e["result"], length_s=e["length_s"], frames=e["frames"], phases=e.get("phases", []),
                  counts=e.get("counts"), holds=len(e.get("holds") or []), steps=e.get("steps"), steps_edit=e.get("steps_edit"), motion_summary=e["motion_summary"], gripper_range_mm=e["gripper_range_mm"],
                  action_state_diff_deg=e["action_state_diff_deg"], recorded=e["recorded"], thumb=_thumb(e), sync=e.get("sync"),
                  depth_sync=e.get("depth_sync"), item=e.get("item", 0), item_name=e.get("item_name"), item_pending=e.get("item_pending"), issues=e.get("issues", []), fail_type=e.get("fail_type", 0)) for e in eps]
    summary = dict(dataset=str(root), name=root.name, step_names=STEP_NAMES, generated=gen, fps=info.get("fps"), features=sorted(info.get("features", {}).keys()),
                   task=eps[-1]["task"] if eps else "", episodes=n, success=ns, failure=nf, unlabeled=nu, success_rate=round(rate, 4),
                   total_minutes=round(tot / 60, 2), check=f"学长标准 {sum(r['status'] == 'ok' for r in A)}/{len(A)}", check_ok=sum(r["status"] == "ok" for r in A), check_total=len(A), checks=chk, items=items)
    (rep / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(rep / "episodes.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["episode", "result", "failure_type", "item", "length_s", "frames", "phases", "motion_summary", "holds", "gripper_range_mm", "action_state_diff_deg", "task", "recorded"])
        for e in eps:
            w.writerow([e["episode"], e["result"], (f'F{e["fail_type"]} {fail["types"].get(str(e["fail_type"]), "")}' if fail and e.get("fail_type") else ""), e.get("item_name", ""), e["length_s"], e["frames"], " | ".join(f'{p["name"]} {p["t0"]}-{p["t1"]}s' for p in e.get("phases", [])),
                        e["motion_summary"], len(e.get("holds") or []), e["gripper_range_mm"], e["action_state_diff_deg"], e["task"], e["recorded"]])
    for e in eps:   # 网页读的单条数据（含修正后的 step）
        (rep / "episodes" / f"view_{e['episode']:04d}.json").write_text(json.dumps(e, ensure_ascii=False), encoding="utf-8")
    res_ok = "failure" if fail else "success"
    if fail:
        ft = {str(e["episode"]): e["fail_type"] for e in eps if e.get("fail_type")}
        summary["fail"] = dict(per_type=int(fail.get("per_type", 20)), types=fail["types"], definitions=fail.get("definitions", {}),
                               count={k: sum(1 for e in eps if str(e.get("fail_type")) == k) for k in fail["types"]},
                               good={k: sum(1 for e in eps if str(e.get("fail_type")) == k and not e.get("issues")) for k in fail["types"]})
        if lab.get("failure_types") != ft or lab.get("failure_names") != fail["types"]:   # 失败类型写进标签文件，导出时一起打包
            lab["failure_types"], lab["failure_names"] = ft, fail["types"]
            save_labels(root.parent, root.name, lab)
    summary["pending_edits"] = sorted(int(k) for k, v in edits.items() if not v.get("applied"))
    summary["labels"] = dict(items=lab["items"], current=lab["current"], per_item=lab["per_item"], task_key=lab.get("task_key"),
                             progress={k: sum(1 for e in eps if str(e.get("item")) == k and e["result"] == res_ok) for k in lab["items"]},
                             good={k: sum(1 for e in eps if str(e.get("item")) == k and e["result"] == res_ok and not e.get("issues")) for k in lab["items"]},
                             issues={k: [e["episode"] for e in eps if str(e.get("item")) == k and e.get("issues")] for k in lab["items"]},
                             has_item_field=ITEM_KEY in info.get("features", {}), item_label=task.get("item_label", "物品"))
    (rep / "labels.json").write_text(json.dumps(lab, ensure_ascii=False, indent=1), encoding="utf-8")
    (rep / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    # report/index.html：单数据集只读版（打包给别人时直接用）；总览在 lerobot_data/index.html
    if APP.exists():
        app = APP.read_text(encoding="utf-8").replace("<script>", f"<script>window.NERO_SINGLE = {json.dumps(root.name)};</script>\n<script>", 1)
        (rep / "index.html").write_text(app, encoding="utf-8")
    for old in rep.glob("ep_*.html"):
        old.unlink()
    write_index(root.parent)
    return summary
