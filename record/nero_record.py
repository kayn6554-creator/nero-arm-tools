#!/usr/bin/env python3
"""lerobot-human-inloop-record 的包装（不改 Evo-RL 源码）：
1) 语音提示 → 醒目横幅 + 响铃；2) 每帧记录定位模式状态（划分大阶段）；3) 每保存一条就截图、统计、刷新报告网页；
4) 数字键 1–4 标 step（名字按任务预设，称重：抓 / 移动 / 放置 / 离开；测水温：抓起 / 到上方 / 测量 / 放回），每帧存进 step_index；
8) 每 0.5 s 把当前阶段、step、物品、定位模式写进 lerobot_data/_live.json，报告网页的“采集控制”读它；网页按钮通过 tmux 把按键送到这个终端。
5) 控制按 30 Hz 跑，数据每 NERO_DECIM（默认 3）帧存 1 帧 → 数据集 10 Hz；
6) 每帧另存主臂夹爪原始值 leader.gripper.pos、两相机时间戳 camera.stamp_ms 和时间差 camera.sync_ms；
7) 两相机深度（对齐到彩色，uint16 毫米）单独存成无损视频 depth/<相机>/episode_XXXXXX.mkv，帧号与数据集对应。
   这些附加字段都不以 observation. 开头，Evo-RL 训练不会把它们当模型输入。"""
import atexit, json, os, re, sys, threading, time
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.expanduser("~"))
import nero_report_lib as NR
TASK = NR.use_task(os.environ.get("NERO_DATASET", "")) or {}     # 按任务预设切换 step 名字
IL = TASK.get("item_label", "物品")
FAIL = NR.fail_cfg(TASK)                  # 失败数据集：只用 f 结束，停在 step k 就是失败类型 Fk
def fail_name(k): return f"F{k} {FAIL['types'].get(str(k), '')}" if FAIL else ""
import nero_depth as ND
import lerobot.scripts.lerobot_record as rec
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_human_inloop_record import main
import lerobot.utils.control_utils as CU

# ================================================================ 交互：提示和按键
# 流程（每条）：准备/复位（不录）→ 按 → 开录 → 1–4 标 step → s/f 结束 → 复位（不录，这时可 ← 丢弃）→ 按 → 保存并开录下一条
C = {"g": "\033[1;97;42m", "y": "\033[1;30;43m", "m": "\033[1;97;45m", "r": "\033[1;97;41m", "b": "\033[1;97;44m", "0": "\033[0m", "d": "\033[2m"}
UI = {"mode": "start", "ep": 0, "need_prep": True, "label": None}
def banner(color, title, lines, bell=True):
    w = 72
    out = (["\a"] if bell else []) + ["", C[color] + f"  {title}".ljust(w) + C["0"]]
    out += [f"  {ln}" for ln in lines]
    print("\n".join(out + [""]), flush=True)

PREP = ["① 主臂把从臂带到物体附近，摆好场景",
        "② 报告网页点「定位开」（或下格 ~/pos_on.sh）→ 网页点物体，看 nav_guard 通过且依次转最低 ≥ 10 cm → Move",
        "③ Target reached 后等 2 秒，点「定位关」（或 ~/pos_off.sh），拖一下主臂确认能动",
        "④ 手放主臂上，上格按 →  开始录制"]
def step_line():
    return "  ".join((f"[{k} {n}]" if k == STEP["cur"] else (f"{k} {n} ✓" if k < STEP["cur"] else f"{k} {n}")) for k, n in NR.STEP_NAMES.items())

def show_prep(first):
    n = UI["ep"] + 1
    banner("y", f"■ 准备第 {n} 条（不录制）", [item_line()] + PREP + [C["d"] + "Esc：结束（什么都不保存）" + C["0"]])

def item_line():
    if not ITEM["cur"]: return f"{IL}：未选（网页上选“当前{IL}”，录完也能在详情里改）"
    lab = ITEM.get("lab") or {}; done = sum(1 for k, v in {**lab.get("applied", {}), **lab.get("episodes", {})}.items() if v == ITEM["cur"])
    return f"{IL}：{ITEM['cur']} · {ITEM['name']}"
def show_rec():
    if FAIL:
        banner("r", f"● 录制中 · 失败数据 · 第 {UI['ep'] + 1} 条", [item_line(), scene_line(), f"step：{step_line()}",
               "按 1–4 切换 step；到哪一步失败，就停在那一步按 f（停在 step k = F k）",
               "  " + "  ".join(fail_name(k) for k in NR.STEP_NAMES), "结束：f 失败      录错了：← 丢弃，回到准备"])
        return
    banner("g", f"● 录制中 · 第 {UI['ep'] + 1} 条", [item_line(), scene_line(), f"step：{step_line()}",
           "按 1–4 切换 step（按错直接按对的）", "完成：s 成功 / f 失败      录错了：← 丢弃，回到准备"])

def show_armed():
    banner("b", f"◎ 就绪 · 第 {UI['ep'] + 1} 条 · 动主臂就开始录（现在不录）", [item_line(), scene_line(),
           f"主臂任一关节动超过 {ARM['deg']:g}° 自动开始录制，开头不会再有静止段",
           C["d"] + "→：不等动作，立刻开始录      Esc：结束（这一条不存）" + C["0"]])

def show_reset():
    n = UI["ep"] + 1; lab = {"success": "成功", "failure": "失败"}.get(UI["label"], "未标注")
    if FAIL and UI["label"] == "failure": lab = "失败 · " + fail_name(STEP["cur"])
    banner("y", f"■ 第 {n} 条已标记「{lab}」，还没保存 · 复位中（不录制）", PREP[:3] + [
           f"④ 手放主臂上，按 →  保存第 {n} 条，并开始录第 {n + 1} 条",
           C["d"] + f"标错了：← 丢弃第 {n} 条      Esc：保存第 {n} 条并结束" + C["0"]])

def nero_say(text, play_sounds=True, blocking=False):
    if text.startswith("Recording episode"):
        UI["ep"] = int(re.findall(r"\d+", text)[0])
    elif text.startswith("Reset the environment"):
        UI["mode"] = "reset"
        if not UI.get("discard"): show_reset()
    elif text.startswith("Re-record"):
        banner("m", f"↺ 第 {UI['ep'] + 1} 条已丢弃，不保存", [], bell=False); UI["need_prep"] = True
    elif text.startswith("Stop recording"):
        banner("r", "■ 录制结束", ["正在保存、编码视频，等程序自己退出，不要按 Ctrl+C"])
    elif text.startswith("Exiting"):
        pass
    else:
        print(f"[提示] {text}", flush=True)
rec.log_say = nero_say

# ---- 物品标签：网页上选“当前物品”，每条开录时读一次 ----
ITEM = {"cur": 0, "name": "", "root": None}
ITEM_FEATURE = {"dtype": "int64", "shape": (1,), "names": None}
def _read_item(ds_root):
    try:
        lab = NR.load_labels(Path(ds_root).parent, Path(ds_root).name)
        ITEM["cur"] = int(lab.get("current") or 0); ITEM["name"] = NR.item_name(lab, ITEM["cur"]); ITEM["lab"] = lab
    except Exception as e:
        ITEM["cur"] = 0; ITEM["name"] = f"读不到标签：{e}"
# ---- 场景（11-09 加）：网页上选“当前场景 + 摆法”，写在 _labels/_scene_current.json；
#      每条开录时读一次，保存时把 {条: 场景} 写进 _labels/<数据集>.scenes.json。不改数据集字段 ----
SCENE = {"cur": None}
SCENE_DIR = Path(os.environ.get("NERO_DATA_ROOT", "/data/panhe/lerobot_data")) / "_labels"
def _read_scene(ds_root):
    try:
        d = json.loads((SCENE_DIR / "_scene_current.json").read_text(encoding="utf-8")).get(Path(ds_root).name)
        SCENE["cur"] = d if isinstance(d, dict) and d.get("scene") else None
    except Exception:
        SCENE["cur"] = None
def scene_line():
    s = SCENE["cur"]
    return f"场景：{s['scene']} · 摆法 {s.get('rot', 0) + 1}" + (f" · {s['desc']}" if s.get("desc") else "") if s else "场景：未选（网页上选“当前场景”）"
def _save_scene(ds_root, ep):
    s = SCENE["cur"]
    if not s: return
    p = SCENE_DIR / f"{Path(ds_root).name}.scenes.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        d = {}
    d.setdefault("episodes", {})[str(int(ep))] = {k: s.get(k) for k in ("scene", "rot", "desc")}
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp"); tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8"); tmp.replace(p)

LIVE = Path(os.environ.get("NERO_DATA_ROOT", "/data/panhe/lerobot_data")) / "_live.json"
if os.environ.get("NERO_DATASET"):
    try:
        LIVE.write_text(json.dumps({"dataset": os.environ["NERO_DATASET"], "started": time.strftime("%Y-%m-%d %H:%M:%S")}), encoding="utf-8")
        NR.write_index(LIVE.parent)
    except Exception:
        pass
LIVE_STOP = threading.Event()
@atexit.register
def _live_off():
    LIVE_STOP.set()
    try:
        LIVE.unlink(missing_ok=True); NR.write_index(LIVE.parent)
    except Exception:
        pass

# ---- 按键：自己处理，只在合适的阶段生效，避免误触 ----
STEP_KEY = NR.STEP_KEY; STEP = {"cur": 1}
STEP_FEATURE = {"dtype": "int64", "shape": (1,), "names": None}
_orig_key = CU.TTYKeyboardListener._handle_key
def _hint(msg): print(f"{C['d']}  （{msg}）{C['0']}", flush=True)
def _key(self, key):
    k = key.lower() if len(key) == 1 else key; ev = self.events; mode = UI["mode"]
    if mode == "busy": return _hint("正在保存上一条，等“✓ 已保存”出现后再按")
    if k == "RIGHT":
        if mode == "rec": return _hint("录制中请用 s / f 结束这一条")
        ev["exit_early"] = True; return
    if k == "LEFT":
        if mode == "prep": return _hint("准备阶段没有可丢弃的，直接按 → 开始")
        if mode == "armed": return _hint("还没开始录，没有可丢弃的；动主臂就开始，或按 Esc 结束")
        ev["rerecord_episode"] = True; ev["exit_early"] = True; return
    if k == "ESC":
        if mode == "rec": ev["rerecord_episode"] = True      # 录制中按 Esc：这一条不保存
        ev["stop_recording"] = True; ev["exit_early"] = True; return
    if k in ("s", "f"):
        if mode != "rec": return _hint("s / f 只在录制中有效")
        if k == "s" and FAIL: return _hint("失败数据集只用 f 结束：停在失败的那一步按 f")
        UI["label"] = "success" if k == "s" else "failure"
        ev["episode_outcome"] = CU.EPISODE_SUCCESS if k == "s" else CU.EPISODE_FAILURE; ev["exit_early"] = True; return
    if len(k) == 1 and k in NR.STEP_NAMES_STR:
        if mode != "rec": return _hint("step 只在录制中有效")
        STEP["cur"] = int(k)
        print(f"\a{C['b']}  step {k} · {NR.STEP_NAMES[STEP['cur']]}  {C['0']}  {step_line()}" + (f"   现在按 f = {fail_name(STEP['cur'])}" if FAIL else ""), flush=True); return
    return _orig_key(self, key)        # i 等其他键交给 Evo-RL
CU.TTYKeyboardListener._handle_key = _key

# ---- 录制循环：第一条和丢弃之后先进“准备”阶段；数据集 10 Hz、循环 30 Hz ----
class _FpsProxy:
    def __init__(self, ds, fps): object.__setattr__(self, "_ds", ds); object.__setattr__(self, "_fps", fps)
    def __getattr__(self, k): return getattr(object.__getattribute__(self, "_ds"), k)
    def __setattr__(self, k, v): setattr(object.__getattribute__(self, "_ds"), k, v)
    @property
    def fps(self): return object.__getattribute__(self, "_fps")
_orig_rl = rec.record_loop
def _record_loop(*a, **kw):
    ds, ev = kw.get("dataset"), kw["events"]
    if ds is None:
        if UI.pop("discard", False): ev["exit_early"] = True   # 录制中按了 ←：跳过复位，直接进“准备”
        return _orig_rl(*a, **kw)
    if UI["need_prep"]:
        _read_item(ds.root); UI["mode"] = "prep"; show_prep(UI["ep"] == 0)
        _orig_rl(*a, **{**kw, "dataset": None, "control_time_s": 36000})
        if ev["stop_recording"]:
            ev["rerecord_episode"] = True; return       # 准备阶段按 Esc：这一条什么都不存
        ev["exit_early"] = False; UI["need_prep"] = False
    _read_item(ds.root); _read_scene(ds.root)
    ev["exit_early"] = False                      # 防止保存期间多按的 → 让这一条一开始就结束
    if ARM["on"]:                                 # 11-09：先“就绪”（不录），主臂动了才开始录，去掉开头的静止段
        UI["mode"] = "armed"; show_armed()
        _wait_motion(a, kw)
        if ev["stop_recording"]:
            ev["rerecord_episode"] = True; return    # 就绪阶段按 Esc：这一条什么都不存
        ev["exit_early"] = False
        _read_item(ds.root); _read_scene(ds.root)    # 就绪期间网页上可能改了物品 / 场景
    UI["mode"] = "rec"; UI["label"] = None; STEP["cur"] = 1; show_rec()
    if ds.fps != kw["fps"]: kw["dataset"] = _FpsProxy(ds, kw["fps"])
    try:
        return _orig_rl(*a, **kw)
    finally:
        UI["mode"] = "reset"
        if ev["rerecord_episode"]: UI["discard"] = True
rec.record_loop = _record_loop

# ---- 动了才开录：就绪阶段照常跑遥操作循环（不存数据），后台线程看主臂关节，动了就结束这段循环 ----
ARM = {"on": os.environ.get("NERO_ARM_START", "1") == "1", "deg": float(os.environ.get("NERO_ARM_DEG", "0.5"))}
def _leader_joints():
    act = DEV.get("act")
    if not isinstance(act, dict): return None
    try:
        v = [float(np.ravel(x)[0]) for k, x in sorted(act.items()) if k.endswith(".pos") and "gripper" not in k]
    except Exception:
        return None
    return np.array(v) if v else None
def _wait_motion(a, kw):
    ev, stop, base = kw["events"], threading.Event(), {"v": None}
    def watch():
        while not stop.wait(0.02):
            v = _leader_joints()
            if v is None: continue
            if base["v"] is None or len(base["v"]) != len(v): base["v"] = v; continue
            if np.degrees(np.abs(v - base["v"]).max()) > ARM["deg"]:
                print(f"{C['g']}  主臂动了 → 开始录制 {C['0']}", flush=True)
                ev["exit_early"] = True; return
    th = threading.Thread(target=watch, daemon=True); th.start()
    try:
        _orig_rl(*a, **{**kw, "dataset": None, "control_time_s": 36000})
    finally:
        stop.set(); th.join(1)

# ---- 订阅定位模式状态（独立 ROS 上下文）----
POS = {"on": False}; _ros = {}
def _pos_listener():
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from rclpy.signals import SignalHandlerOptions
    from std_msgs.msg import Bool
    ctx = Context(); rclpy.init(context=ctx, signal_handler_options=SignalHandlerOptions.NO)
    n = Node("nero_record_phase", context=ctx)
    n.create_subscription(Bool, "/nero/positioning_state", lambda m: POS.__setitem__("on", bool(m.data)),
                          QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    ex = SingleThreadedExecutor(context=ctx); ex.add_node(n)
    _ros.update(ctx=ctx, node=n, ex=ex)
    ex.spin()
_th = threading.Thread(target=_pos_listener, daemon=True); _th.start()
@atexit.register
def _stop_ros():
    try:
        import rclpy
        if "ex" in _ros: _ros["ex"].shutdown()
        _th.join(2)
        if "node" in _ros: _ros["node"].destroy_node()
        if "ctx" in _ros: rclpy.shutdown(context=_ros["ctx"])
    except Exception:
        pass

# ---- 降采样、主臂夹爪、相机时间戳 ----
DECIM = int(os.environ.get("NERO_DECIM", "3"))
CAMS = ["front", "wrist"]
EXTRA = {"leader.gripper.pos": {"dtype": "float32", "shape": (1,), "names": None},
         "camera.stamp_ms": {"dtype": "float64", "shape": (len(CAMS),), "names": CAMS},
         "camera.sync_ms": {"dtype": "float32", "shape": (1,), "names": None},
         "camera.depth_rgb_ms": {"dtype": "float32", "shape": (len(CAMS),), "names": CAMS}}   # 每个相机 深度−彩色 时间戳差
DEV = {"leader": None, "follower": None, "grip": float("nan")}
from lerobot_teleoperator_nero import NeroLeader
from lerobot_robot_nero import NeroFollower
_orig_get = NeroLeader.get_action
def _get_with_grip(self):
    DEV["leader"] = self; act = _orig_get(self); DEV["act"] = act
    try:
        g = self._gripper.get_gripper_ctrl_states() if self._gripper is not None else None
        if g is not None: DEV["grip"] = float(g.msg.value)
    except Exception:
        pass
    return act
NeroLeader.get_action = _get_with_grip
_orig_fconn = NeroFollower.connect
def _fconn(self, *a, **kw):
    DEV["follower"] = self; return _orig_fconn(self, *a, **kw)
NeroFollower.connect = _fconn

def _extras():
    fr = getattr(DEV["follower"], "cam_frames", None) or {}
    st = np.array([fr[c].stamp_ms if c in fr else np.nan for c in CAMS], dtype=np.float64)
    dd = np.array([(fr[c].depth_stamp_ms - fr[c].stamp_ms) if c in fr and getattr(fr[c], "depth_stamp_ms", None) is not None else np.nan for c in CAMS], dtype=np.float32)
    return {"leader.gripper.pos": np.array([DEV["grip"]], dtype=np.float32), "camera.stamp_ms": st,
            "camera.sync_ms": np.array([st[1] - st[0]], dtype=np.float32), "camera.depth_rgb_ms": dd}
def _extra_in(features):   # 老数据集没有新加的字段：只写数据集里有的
    return {k: v for k, v in EXTRA.items() if k in features}

_orig_create = LeRobotDataset.create.__func__
def _create_with_step(cls, repo_id, fps, *a, **kw):
    if fps % DECIM: raise ValueError(f"录制帧率 {fps} 不能被 NERO_DECIM={DECIM} 整除")
    kw["features"] = {**kw["features"], STEP_KEY: STEP_FEATURE, NR.ITEM_KEY: ITEM_FEATURE, **EXTRA}
    print(f"[数据] 控制 {fps} Hz，存储 {fps // DECIM} Hz（每 {DECIM} 帧存 1 帧）", flush=True)
    return _orig_create(cls, repo_id, fps // DECIM, *a, **kw)
LeRobotDataset.create = classmethod(_create_with_step)
_orig_sanity = rec.sanity_check_dataset_robot_compatibility
def _sanity_with_step(dataset, robot, fps, features):
    if STEP_KEY in dataset.features: features = {**features, STEP_KEY: STEP_FEATURE}
    if NR.ITEM_KEY in dataset.features: features = {**features, NR.ITEM_KEY: ITEM_FEATURE}
    features = {**features, **_extra_in(dataset.features)}
    if fps == dataset.fps * DECIM: fps = dataset.fps
    return _orig_sanity(dataset, robot, fps, features)
rec.sanity_check_dataset_robot_compatibility = _sanity_with_step

# ---- 深度 ----
DEPTH = {"on": os.environ.get("NERO_DEPTH", "1") == "1", "rec": None}
_orig_clear = LeRobotDataset.clear_episode_buffer
def _clear_with_depth(self, *a, **kw):
    if DEPTH["rec"] is not None: DEPTH["rec"].abort()      # ← 重录：丢掉这一条的深度
    return _orig_clear(self, *a, **kw)
LeRobotDataset.clear_episode_buffer = _clear_with_depth
@atexit.register
def _depth_exit():
    if DEPTH["rec"] is not None: DEPTH["rec"].abort()

# ---- 每帧记录定位状态和 step ----
FLAGS = []
_orig_add = LeRobotDataset.add_frame
TICK = {"n": 0}
def _add_with_phase(self, frame):
    if not self.episode_buffer or self.episode_buffer.get("size", 0) == 0:   # 新的一条（第 1 帧一定会存，所以 size 只在开头为 0）
        FLAGS.clear(); TICK["n"] = 0
    TICK["n"] += 1
    if (TICK["n"] - 1) % DECIM:          # 降采样：只存每 DECIM 帧里的第 1 帧
        return None
    FLAGS.append(POS["on"])
    if STEP_KEY in self.features:
        frame = {**frame, STEP_KEY: np.array([STEP["cur"]], dtype=np.int64)}
    if NR.ITEM_KEY in self.features:
        frame = {**frame, NR.ITEM_KEY: np.array([ITEM["cur"]], dtype=np.int64)}
    ex = _extra_in(self.features)
    if ex:
        frame = {**frame, **{k: v for k, v in _extras().items() if k in ex}}
    if DEPTH["on"]:
        if DEPTH["rec"] is None: DEPTH["rec"] = ND.DepthRecorder(self.root, CAMS, self.fps)
        if not self.episode_buffer or self.episode_buffer.get("size", 0) == 0: DEPTH["rec"].start()
        fr = getattr(DEV["follower"], "cam_frames", None) or {}
        DEPTH["rec"].add({c: fr[c].depth for c in CAMS if c in fr})
    return _orig_add(self, frame)
LeRobotDataset.add_frame = _add_with_phase

# ---- 每保存一条：截图 + 统计 → 保存 → 刷新报告 ----
_orig_save = LeRobotDataset.save_episode
def _save_with_report(self, *a, **kw):
    if not self.episode_buffer or self.episode_buffer.get("size", 0) == 0:
        print(f"{C['m']}  这一条没有录到任何帧，跳过保存 {C['0']}", flush=True); return None
    UI["mode"] = "busy"
    try:
        return _save_inner(self, *a, **kw)
    finally:
        UI["mode"] = "reset"
def _save_inner(self, *a, **kw):
    try:
        if self.episode_buffer and self.episode_buffer.get("size", 0) > 0:
            self._wait_image_writer()
            NR.capture_live_episode(self, kw.get("extra_episode_metadata"), list(FLAGS))
    except Exception as e:
        print(f"[报告] 本条统计失败（不影响录制）：{e}", flush=True)
    try:   # 深度先收尾（save_episode 内部会清空缓冲区）
        b = self.episode_buffer
        if DEPTH["rec"] is not None and b and b.get("size", 0) > 0:
            r = DEPTH["rec"].finish(int(np.ravel(b["episode_index"])[0]), int(b["size"]))
            DEPTH["msg"] = "深度 " + " / ".join(f"{c} {v['frames']} 帧{'' if v['ok'] else ' ✗ ' + str(v['error'] or '帧数不符')}" for c, v in r.items())
    except Exception as e:
        print(f"[深度] 保存失败：{e}", flush=True)
    b = self.episode_buffer
    ep_idx = int(np.ravel(b["episode_index"])[0]) if b and b.get("size", 0) > 0 else None
    res = _orig_save(self, *a, **kw)
    try:
        if ep_idx is not None: _save_scene(self.root, ep_idx)
    except Exception as e:
        print(f"[场景] 记录失败（不影响录制）：{e}", flush=True)
    try:
        s = NR.build_site(self.root)
        print(f"{C['g']}  ✓ 已保存第 {s['episodes']} 条 {C['0']}  {IL} {ITEM['name'] or '未选'} · {DEPTH.pop('msg', '')} · 共 {s['episodes']} 条 · {s['check']} · 报告已更新", flush=True)
    except Exception as e:
        print(f"[报告] 网页更新失败（不影响录制）：{e}", flush=True)
    return res
LeRobotDataset.save_episode = _save_with_report

# ---- 每 0.5 s 写一次当前状态，给报告网页的“采集控制”看 ----
def _live_loop():
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    while not LIVE_STOP.wait(0.5):
        try:
            d = dict(dataset=os.environ["NERO_DATASET"], started=started, mode=UI["mode"], ep=UI["ep"] + 1, label=UI["label"],
                     step=STEP["cur"], step_name=NR.STEP_NAMES.get(STEP["cur"], ""), steps=NR.STEP_NAMES, item=ITEM["cur"],
                     item_name=ITEM.get("name", ""), item_label=IL, positioning=POS["on"], t=time.time(),
                     fail=bool(FAIL), fail_types=(FAIL or {}).get("types"), scene=SCENE["cur"], arm_start=ARM["on"])
            if LIVE_STOP.is_set(): break
            tmp = LIVE.with_suffix(".tmp"); tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8"); tmp.replace(LIVE)
        except Exception:
            pass
if os.environ.get("NERO_DATASET"):
    threading.Thread(target=_live_loop, daemon=True).start()


sys.exit(main())

