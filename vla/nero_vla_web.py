#!/usr/bin/env python3
"""NERO 采集 + 模型测试台（网页）：把 nero_vla_client.py / nero_replay_chunk.py 和急停、关锁、夹爪做成按钮。

11-09 起合并了采集：顶部切换「采集 / 测试」。采集模式不自己控制录制，只把按钮转发给报告服务
（~/nero_report_server.py，默认 http://127.0.0.1:8766 的 /api/ctl、/api/ctl_state、/api/labels），
录制程序仍在 tmux 里由报告服务送按键。场景表在 ~/nero_scenes.json，当前场景写在
<数据根>/_labels/_scene_current.json，录制程序每条开录时读它。

启动（tmux 新窗口，系统 Python，不用 conda）：
    conda deactivate
    /usr/bin/python3 ~/nero_vla_web.py            # 默认 127.0.0.1:8767
Windows：ssh -L 8767:127.0.0.1:8767 panhe@192.168.1.102，浏览器打开 http://127.0.0.1:8767

每个按钮都在干净环境里启动对应命令（evo-rl 的命令先 source ~/env_evo.sh，系统 Python 的命令 source ROS + agx_arm_ws），
同一时间只跑一个任务；急停、关锁、停止随时可按。
"""
import json
import os
import re
import shlex
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from glob import glob
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

HOME = str(Path.home())
RUNS = os.environ.get("NERO_VLA_RUNS", "/data/panhe/vla_runs")
EVALS = os.environ.get("NERO_VLA_EVAL_DIR", "/data/panhe/vla_eval")   # 人工评测记录：<任务>.jsonl，一行一次运行
DATA = os.environ.get("NERO_DATA_ROOT", "/data/panhe/lerobot_data")
EVAL_TARGET = 20
PORT = int(os.environ.get("NERO_VLA_WEB_PORT", "8767"))
FAKE = os.environ.get("NERO_VLA_WEB_FAKE") == "1"   # 本地调试：命令换成 echo

EVO = "source ~/miniconda3/etc/profile.d/conda.sh && source ~/env_evo.sh 2>/dev/null; "
SYS = ("export PYTHONPATH=/data/kaiz/projects/NERO-arm/pyAgxArm; source /opt/ros/humble/setup.bash; "
       "source /data/kaiz/projects/NERO-arm/agx_arm_ws/install/setup.bash 2>/dev/null; ")
REPORT = os.environ.get("NERO_REPORT_URL", "http://127.0.0.1:8766")   # 报告服务（采集控制）


def load_all_tasks():
    try:
        return json.loads(Path(HOME, "nero_tasks.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_task_names():
    """测试用的任务：nero_tasks.json 里不是失败数据集的预设。启动时读一次，改了 json 要重启网页。"""
    out = {k: t.get("name", k) for k, t in load_all_tasks().items() if t.get("task") and not t.get("failure")}
    return out or {"weigh": "称重", "task2": "水温测量", "task3": "风速测量"}


TASKS = load_task_names()

# ---------------------------------------------------------------- 周末实验：边跑边标步骤
STEP_FILE = Path(HOME, ".nero_vla_step")   # 客户端每次推理前读它（--step-file）；“本步完成”时 +1
# 示范数据里每步时长的中位数（秒，11-06 统计）：测试台用它提示“这一步拖太久了”
DEMO_STEP_S = {"weigh": {1: 8.9, 2: 4.6, 3: 2.3, 4: 2.5}, "task2": {1: 9.4, 2: 6.3, 3: 16.2, 4: 9.0},
               "task3": {1: 8.7, 2: 3.5, 3: 24.8, 4: 6.2}}
FAIL_KINDS = {1: ["停在上方", "提前合爪", "到位不合爪", "夹偏", "没夹住", "搬运掉落"],
              2: ["对位偏", "途中掉落", "中途停滞", "阶段混淆（折返/跳步）"],
              3: ["卡在测量/放置", "放置偏", "深度不够", "对位偏"],
              4: ["复位折返", "夹爪未松开", "放偏", "中途停滞"]}
FAIL_COMMON = ["碰撞", "安全停止（按了实体急停）", "操作失误（步骤键按错/按晚，不算模型）", "其它（写备注）"]
OPERATOR_ERROR = "操作失误"
trial = {"active": False}

# ---------------------------------------------------------------- 11-07：模型预设、连接检查、交替顺序
# 预设 = 一个模型在真机上该用的全部设置。变体名是预设名时，④ 的设置必须和预设一致才能开跑。
PRESET_FILE = Path(HOME, "nero_vla_presets.json")
DEFAULT_PRESETS = [
    {"name": "M0", "host": "172.16.11.115", "port": 6699, "step_prompt": False, "exec_steps": 30, "slow": 2,
     "gripper_mode": "delta", "note": "学长基线 epoch3（= 周六的 E0）"},
    {"name": "M1", "host": "172.16.11.113", "port": 6707, "step_prompt": True, "exec_steps": 30, "slow": 2,
     "gripper_mode": "delta", "note": "+ 步骤文本"},
    {"name": "M2", "host": "172.16.11.113", "port": 6706, "step_prompt": True, "exec_steps": 30, "slow": 2,
     "gripper_mode": "absolute", "note": "M1 + 三处数据修正"},
    {"name": "M2b", "host": "172.16.11.113", "port": 6707, "step_prompt": False, "exec_steps": 30, "slow": 2,
     "gripper_mode": "absolute", "note": "M2 去掉步骤文本（和 M1 共用 7 号卡，换模型后用）"},
    {"name": "M2+E1", "host": "172.16.11.113", "port": 6706, "step_prompt": True, "exec_steps": 10, "slow": 2,
     "gripper_mode": "absolute", "note": "M2，每块只执行 10 步"},
]
pings = {}   # "host:port" -> {t, ok, md / err}


def load_presets():
    try:
        p = json.loads(PRESET_FILE.read_text(encoding="utf-8"))
        if isinstance(p, list) and p:
            return p
    except Exception:
        pass
    return DEFAULT_PRESETS


def save_presets(body):
    rows = body.get("presets")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 20:
        raise ValueError("预设格式不对")
    out = []
    for r in rows:
        name = re.sub(r"[^\w.+-]", "", str(r.get("name", "")))[:24]
        if not name:
            raise ValueError("预设要有名字")
        out.append({"name": name, "host": host_of(r), "port": num(r, "port", 1, 65535, int),
                    "step_prompt": bool(r.get("step_prompt")), "exec_steps": num(r, "exec_steps", 1, 30, int),
                    "slow": num(r, "slow", 1, 10), "gripper_mode": "absolute" if r.get("gripper_mode") == "absolute" else "delta",
                    "note": str(r.get("note", ""))[:80]})
    if len({r["name"] for r in out}) != len(out):
        raise ValueError("预设名字重复了")
    PRESET_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return f"已保存 {len(out)} 个预设"


def host_of(p):
    h = str(p.get("host", "172.16.11.115")).strip()
    if not re.fullmatch(r"[A-Za-z0-9.-]{1,64}", h):
        raise ValueError("服务地址只能是 IP 或主机名")
    return h


def check_preset(variant, host, port, sp, e, s):
    """变体名是某个预设的名字时，设置必须和预设一致（防止逐次交替时配错）。"""
    pr = next((x for x in load_presets() if x["name"] == variant), None)
    if not pr:
        return None
    bad = [f"{k} 应为 {want}（现在 {got}）" for k, want, got in
           (("服务地址", pr["host"], host), ("端口", pr["port"], port), ("步骤文本", "开" if pr["step_prompt"] else "关", "开" if sp else "关"),
            ("每块执行", pr["exec_steps"], e), ("放慢", pr["slow"], s)) if str(want) != str(got) and not (k == "放慢" and float(want) == float(got))]
    if bad:
        raise ValueError(f"变体 {variant} 的设置和预设不一致：" + "；".join(bad) + "。点 ④ 里的预设按钮重新套用。")
    return pr


def sched_path():
    return Path(EVALS, "schedule.json")


def load_sched():
    try:
        return json.loads(sched_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_sched(body):
    task = task_of(body)
    order = [re.sub(r"[^\w.+-]", "", str(x))[:24] for x in (body.get("order") or [])]
    order = [x for x in order if x][:300]
    s = load_sched()
    if order:
        s[task] = {"order": order, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    else:
        s.pop(task, None)
    sched_path().parent.mkdir(parents=True, exist_ok=True)
    sched_path().write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    return f"{TASKS[task]}：顺序已保存（{len(order)} 次）" if order else f"{TASKS[task]}：顺序已清空"


def counted(r):
    return not str(r.get("fail_kind", "")).startswith(OPERATOR_ERROR)


def plan_state():
    """每个任务：下一次该跑哪个变体；每个变体的进度。操作失误的记录不计数（要重跑）。"""
    sched, out = load_sched(), {}
    for task in TASKS:
        rows = [r for r in load_evals(task) if counted(r)]
        sc = sched.get(task)
        nxt = None
        if sc:
            order, created = sc["order"], sc["created"]
            done = sum(1 for r in rows if r.get("variant") in set(order) and (r.get("saved") or "") >= created)
            nxt = {"variant": order[done] if done < len(order) else None, "done": done, "total": len(order)}
        per = {}
        for r in rows:
            d = per.setdefault(r.get("variant") or "-", {"n": 0, "success": 0, "s1": 0, "reach": [0, 0, 0, 0]})
            d["n"] += 1
            d["success"] += bool(r.get("success"))
            d["s1"] += r["steps"].get("1") == "done"
            for i in range(4):
                d["reach"][i] += r["steps"].get(str(i + 1)) == "done"
        if sc:
            for vname in sc["order"]:
                per.setdefault(vname, {"n": 0, "success": 0, "s1": 0, "reach": [0, 0, 0, 0]})
                per[vname]["target"] = sc["order"].count(vname)
        out[task] = {"schedule": sc, "next": nxt, "per_variant": per}
    return out


def write_step(k: int):
    tmp = STEP_FILE.with_suffix(".tmp")
    tmp.write_text(f"{k}\n")
    tmp.replace(STEP_FILE)


def trial_start(task, variant, item, step_prompt, server="", fan_pos="", scene=""):
    trial.clear()
    trial.update(active=True, task=task, variant=variant, item=item, step_prompt=step_prompt, step=1, server=server, fan_pos=fan_pos, scene=scene,
                 t0=time.time(), step_t0=time.time(), events=[], result=None, saved=False)
    write_step(1)


def trial_event(op, body):
    """op: step_done / fail / undo / note。成功或失败后停止运行并自动存评测。"""
    if not trial.get("active"):
        raise ValueError("当前没有在评测的运行（先在 ④ 开始自动运行）")
    now = time.time()
    if op == "undo":
        if not trial["events"] or now - trial["events"][-1]["t"] > 8 or trial["result"]:
            raise ValueError("只能撤销 8 秒内的“本步完成”")
        e = trial["events"].pop()
        trial["step"], trial["step_t0"] = e["step"], e["step_t0"]
        write_step(trial["step"])
        return f"已撤销，回到第 {trial['step']} 步"
    if op == "note":
        trial["note"] = str(body.get("note", ""))[:300]
        return "备注已记下"
    if trial["result"]:
        raise ValueError("这次已经结束")
    k = trial["step"]
    if op == "step_done":
        trial["events"].append({"op": "done", "step": k, "t": now, "step_t0": trial["step_t0"], "dur": round(now - trial["step_t0"], 1)})
        if k >= 4:
            trial["result"] = "success"
            stop_job("4 步全部完成")
            return "成功：4 步全部完成"
        trial["step"], trial["step_t0"] = k + 1, now
        write_step(k + 1)
        return f"第 {k} 步完成，进入第 {k + 1} 步"
    if op == "fail":
        kind = str(body.get("kind", ""))[:40]
        trial["events"].append({"op": "fail", "step": k, "t": now, "kind": kind, "dur": round(now - trial["step_t0"], 1)})
        trial["result"] = "fail"
        trial["fail_kind"] = kind
        stop_job(f"第 {k} 步失败：{kind}")
        return f"第 {k} 步失败（{kind}），已停止"
    raise ValueError("未知操作")


def trial_finish():
    """自动运行结束时调用：标了成功/失败就自动写评测记录和 events.json；没标的留给 ⑤ 事后评测。"""
    if not trial.get("active"):
        return
    trial["active"] = False
    run_dir = pending_meta.get("run_dir")
    if not trial.get("result"):
        add_log("── 这次没有标成功/失败，留到 ⑤ 事后评测")
        return
    if not run_dir:
        add_log("── 没拿到这次运行的目录（客户端可能没启动成功），评测没有自动保存")
        return
    run = Path(run_dir).parent.name
    k = trial["step"]
    steps = {str(i): ("done" if i < k or trial["result"] == "success" else "fail" if i == k else "na") for i in range(1, 5)}
    times = {str(e["step"]): e["dur"] for e in trial["events"]}
    try:
        Path(run_dir, "events.json").write_text(json.dumps({k2: trial.get(k2) for k2 in
            ("task", "variant", "item", "step_prompt", "server", "fan_pos", "scene", "t0", "events", "result", "fail_kind", "note")},
            ensure_ascii=False, indent=1), encoding="utf-8")
        save_eval({"task": trial["task"], "run": run, "steps": steps, "success": trial["result"] == "success",
                   "item": trial["item"], "video": "", "note": trial.get("note", ""),
                   "variant": trial["variant"], "fail_kind": trial.get("fail_kind", ""), "step_s": times,
                   "step_prompt": trial["step_prompt"], "server": trial.get("server", ""), "fan_pos": trial.get("fan_pos", ""),
                   "scene": trial.get("scene", "")})
        trial["saved"] = True
        add_log(f"── 评测已自动保存：{run} {trial['variant']} {'成功' if trial['result'] == 'success' else '失败 第 %d 步 %s' % (k, trial.get('fail_kind'))}")
    except Exception as e:
        add_log(f"（自动保存评测失败：{e}，请在 ⑤ 手动评测）")


def trial_state():
    st = {k: trial.get(k) for k in ("active", "task", "variant", "item", "step_prompt", "server", "fan_pos", "scene", "step", "result", "fail_kind", "saved")}
    if trial.get("task"):
        info = task_info(trial["task"])
        st["task_name"], st["step_names"] = info["name"], [info["steps"].get(str(i), str(i)) for i in range(1, 5)]
    if trial.get("t0"):
        now = time.time() if trial.get("active") else (trial["events"][-1]["t"] if trial.get("events") else time.time())
        st["step_elapsed"] = round(now - trial["step_t0"], 1) if trial.get("active") else None
        st["total_elapsed"] = round(now - trial["t0"], 1)
        st["demo_step_s"] = DEMO_STEP_S.get(trial.get("task"), {}).get(trial.get("step"))
        st["events"] = trial["events"]
    run_dir = pending_meta.get("run_dir")
    if trial.get("t0") and run_dir:
        try:
            st["live"] = json.loads(Path(run_dir, "live_status.json").read_text(encoding="utf-8"))
        except Exception:
            st["live"] = None
    return st


lock = threading.Lock()
log = deque(maxlen=1500)
job = {"name": None, "proc": None, "rc": None, "started": 0.0, "ended": 0.0}
pending_meta = {}     # 下一次 live 的物品 / 视频编号 / 是否结束后回到主臂姿态，看到 RUN_DIR= 时写进那次的记录
last_estop = [0.0]    # 最近一次按急停的时刻：急停后不自动动手臂


def add_log(line: str):
    if line.startswith(("PING=", "PING_FAIL=")):   # nero_vla_ping.py 的结果：记下来给 ④ 显示
        try:
            j = json.loads(line.split("=", 1)[1])
            pings[f"{j['host']}:{j['port']}"] = {"t": time.time(), "ok": line.startswith("PING="), "md": j.get("md"), "err": j.get("err")}
            line = (f"── 连接 {j['host']}:{j['port']} 正常  metadata={json.dumps(j.get('md'), ensure_ascii=False)}" if line.startswith("PING=")
                    else f"── 连接 {j['host']}:{j['port']} 失败：{j.get('err')}")
        except Exception:
            pass
    log.append(time.strftime("%H:%M:%S ") + line if line.startswith("──") else line)
    if line.startswith("RUN_DIR=") and pending_meta:
        try:
            d = Path(line.split("=", 1)[1].strip())
            if str(d).startswith(RUNS + "/"):
                (d / "meta_web.json").write_text(json.dumps(pending_meta, ensure_ascii=False), encoding="utf-8")
                pending_meta["run_dir"] = str(d)
        except Exception as e:
            log.append(f"（写运行标签失败：{e}）")


def spawn(script: str, stdin_text: str = ""):
    env = {"HOME": HOME, "USER": os.environ.get("USER", "panhe"), "LANG": "C.UTF-8",
           "PATH": "/usr/local/bin:/usr/bin:/bin", "PYTHONUNBUFFERED": "1", "TERM": "dumb"}
    if FAKE:
        script = "echo [FAKE] " + shlex.quote(script) + "; sleep 2; echo 完成"
    p = subprocess.Popen(["bash", "-c", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, env=env, cwd=HOME, start_new_session=True)
    if stdin_text:
        p.stdin.write(stdin_text.encode())
        p.stdin.flush()
    return p


def pump(p, on_exit=None):
    buf = b""
    while True:
        ch = p.stdout.read1(4096) if hasattr(p.stdout, "read1") else p.stdout.read(1)
        if not ch:
            break
        buf += ch
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            add_log(line.decode("utf-8", "replace").rstrip("\r"))
    if buf:
        add_log(buf.decode("utf-8", "replace"))
    rc = p.wait()
    if on_exit:
        on_exit(rc)
    return rc


def start_job(name: str, script: str, stdin_text: str = "", on_done=None):
    with lock:
        if job["proc"] is not None and job["proc"].poll() is None:
            return False, f"已有任务在运行：{job['name']}"
        add_log(f"── 开始：{name}")
        p = spawn(script, stdin_text)
        job.update(name=name, proc=p, rc=None, started=time.time(), ended=0.0)

    def done(rc):
        with lock:
            job.update(rc=rc, ended=time.time())
        add_log(f"── 结束：{name}（返回码 {rc}）")
        if on_done:
            on_done(rc)

    threading.Thread(target=pump, args=(p, done), daemon=True).start()
    return True, "已启动"


def run_side(name: str, script: str):
    """急停、关锁：不占任务位，立即执行。"""
    add_log(f"── {name}")
    p = spawn(script)
    threading.Thread(target=pump, args=(p, lambda rc: add_log(f"── {name} 完成（{rc}）")), daemon=True).start()


def stop_job(reason="停止"):
    with lock:
        p = job["proc"]
    if p is not None and p.poll() is None:
        add_log(f"── {reason}：向当前任务发送 Ctrl+C（脚本会关锁 + 原地保持）")
        try:
            os.killpg(p.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        return True
    return False


# ---------------------------------------------------------------- 11-09：采集（转发给报告服务）和场景
SCENES_FILE = Path(HOME, "nero_scenes.json")
SCENE_CUR = Path(DATA, "_labels", "_scene_current.json")
scene_lock = threading.Lock()


def report_call(path, body=None, timeout=4):
    """调报告服务；它出错时回 400 + {ok:false,error}，这里统一变成 ValueError。"""
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(REPORT + path, data=data, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            j = json.loads(e.read())
        except Exception:
            j = {}
        raise ValueError(j.get("error") or f"报告服务返回 {e.code}")
    except Exception as e:
        raise ValueError(f"连不上报告服务 {REPORT}（~/nero_report_serve.sh 在运行吗？）：{e}")


def load_scenes():
    try:
        return json.loads(SCENES_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"slots": ["P1", "P2", "P3", "P4"], "rotate_every": 5, "tasks": {}}


def preset_of(key):
    t = load_all_tasks().get(key)
    if not t or not t.get("dataset"):
        raise ValueError(f"没有这个任务预设：{key}")
    return t


def scene_task_key(key):
    """失败数据集用对应成功数据集的场景表（items_from 指向的那个任务）。"""
    t = preset_of(key)
    if t.get("items_from"):
        for k, x in load_all_tasks().items():
            if x.get("dataset") == t["items_from"]:
                return k
    return key


def layout(sc, rot, slots):
    return [{"tool": tl, "slot": slots[(i + rot) % len(slots)]} for i, tl in enumerate(sc.get("tools") or [])]


def scene_counts(dataset):
    """{场景: {good: 成功且没问题的条数, all: 条数}}，按 _labels/<数据集>.scenes.json 和报告的 summary.json 对上。"""
    if not re.fullmatch(r"[\w.\-]+", dataset):
        return {}
    try:
        eps = json.loads(Path(DATA, "_labels", f"{dataset}.scenes.json").read_text(encoding="utf-8")).get("episodes", {})
    except Exception:
        eps = {}
    try:
        items = json.loads(Path(DATA, dataset, "report", "summary.json").read_text(encoding="utf-8")).get("items", [])
    except Exception:
        items = []
    fail = bool(preset_of_dataset(dataset).get("failure"))
    res = {str(e.get("episode")): e for e in items}
    out = {}
    for ep, s in eps.items():
        d = out.setdefault(str(s.get("scene")), {"good": 0, "all": 0})
        e = res.get(str(ep))
        if e is None:      # 报告里已经没有这条（删了）
            continue
        d["all"] += 1
        d["good"] += e.get("result") == ("failure" if fail else "success") and not e.get("issues")
    return out


def preset_of_dataset(dataset):
    return next((t for t in load_all_tasks().values() if t.get("dataset") == dataset), {})


def read_scene_cur():
    try:
        return json.loads(SCENE_CUR.read_text(encoding="utf-8"))
    except Exception:
        return {}


def scene_state(key):
    t, S = preset_of(key), load_scenes()
    slots, every = S.get("slots") or ["P1", "P2", "P3", "P4"], int(S.get("rotate_every") or 5)
    tk = scene_task_key(key)
    scenes = (S.get("tasks", {}).get(tk) or {}).get("scenes", [])
    counts = scene_counts(t["dataset"])
    rows = []
    for sc in scenes:
        c = counts.get(sc["id"], {"good": 0, "all": 0})
        auto = (c["all"] // every) % len(slots)
        rows.append({**sc, "good": c["good"], "all": c["all"], "auto_rot": auto, "layout": layout(sc, auto, slots)})
    cur = read_scene_cur().get(t["dataset"])
    if cur:
        sc = next((x for x in scenes if x["id"] == cur.get("scene")), None)
        cur = {**cur, "layout": layout(sc, int(cur.get("rot") or 0), slots) if sc else [], "fixed": (sc or {}).get("fixed", []),
               "name": (sc or {}).get("name", ""), "pair": (sc or {}).get("pair", [])}
    return {"preset": key, "dataset": t["dataset"], "slots": slots, "rotate_every": every, "scenes": rows, "current": cur,
            "total": (S.get("tasks", {}).get(tk) or {}).get("total"), "unknown": {k: v for k, v in counts.items() if k not in {s["id"] for s in scenes}}}


def set_scene(body):
    key = str(body.get("preset", ""))
    t, st = preset_of(key), scene_state(key)
    sid = str(body.get("scene", ""))
    with scene_lock:
        cur = read_scene_cur()
        if not sid:
            cur.pop(t["dataset"], None)
            msg = "已清除当前场景"
        else:
            sc = next((x for x in st["scenes"] if x["id"] == sid), None)
            if not sc:
                raise ValueError(f"场景表里没有 {sid}")
            rot = sc["auto_rot"] if str(body.get("rot", "auto")) == "auto" else int(body["rot"]) % len(st["slots"])
            lay = layout(sc, rot, st["slots"])
            desc = "、".join(f"{x['tool']}@{x['slot']}" for x in lay)
            cur[t["dataset"]] = {"scene": sid, "rot": rot, "desc": desc, "set": time.strftime("%Y-%m-%d %H:%M:%S")}
            msg = f"当前场景 {sid} · 摆法 {rot + 1}：{desc or '（没有要轮换的工具）'}。下一条开录时生效"
        SCENE_CUR.parent.mkdir(parents=True, exist_ok=True)
        tmp = SCENE_CUR.with_suffix(".tmp")
        tmp.write_text(json.dumps(cur, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(SCENE_CUR)
    return msg


def collect_state(key):
    """一次拿齐采集页要的东西：录制状态（报告服务）、场景、物品名单。正在录的数据集优先于下拉框里选的。"""
    out = {"report_url": REPORT}
    try:
        out["ctl"] = report_call("/api/ctl_state", timeout=3)
    except ValueError as e:
        out["ctl"], out["ctl_error"] = None, str(e)
    live = (out.get("ctl") or {}).get("live") or {}
    if live.get("dataset") and live.get("fresh"):
        key = next((k for k, t in load_all_tasks().items() if t.get("dataset") == live["dataset"]), key)
    out["preset"] = key
    try:
        out["scene"] = scene_state(key)
        ds = out["scene"]["dataset"]
        lab = json.loads(Path(DATA, "_labels", f"{ds}.json").read_text(encoding="utf-8"))
        out["items"], out["item_cur"] = lab.get("items", {}), lab.get("current")
    except ValueError as e:
        out["scene_error"] = str(e)
    except Exception:     # 新数据集还没有标签文件：和报告服务一样，按 item_slots 给空名字
        n = int(load_all_tasks().get(key, {}).get("item_slots") or 0)
        out["items"], out["item_cur"] = {str(i): "" for i in range(1, n + 1)}, (1 if n else 0)
    return out


# ---------------------------------------------------------------- 参数检查
def num(params, key, lo, hi, cast=float):
    v = cast(params[key])
    if not (lo <= v <= hi):
        raise ValueError(f"{key} 应在 {lo}~{hi}")
    return v


def task_of(params):
    t = params.get("task", "weigh")
    if t not in TASKS:
        raise ValueError("未知任务")
    return t


def chunk_file(path):
    real = os.path.realpath(path)
    if not real.startswith(RUNS + "/") or not real.endswith("chunks.npz") or not os.path.exists(real):
        raise ValueError("动作文件不在 vla_runs 里")
    return real


def handle_action(a: str, p: dict):
    if a == "estop":
        last_estop[0] = time.time()
        stop_job("急停")
        run_side("软件急停（关锁 + 原地保持）", SYS +
                 'ros2 service call /control_enable std_srvs/srv/SetBool "{data: false}"; '
                 "ros2 service call /emergency_stop std_srvs/srv/Empty")
        return "已发送急停"
    if a == "lock":
        run_side("关控制锁", SYS + "~/nero_lock.sh")
        return "已发送关锁"
    if a == "stop":
        return "已发送 Ctrl+C" if stop_job() else "当前没有任务"

    port = num(p, "port", 1, 65535, int) if "port" in p else 6699
    host = host_of(p)
    task = task_of(p)
    if a == "ping":   # 只连模型服务读 metadata，不读相机、不动手臂，不占任务位
        run_side(f"检查连接 {host}:{port}", EVO + f"python -u ~/nero_vla_ping.py {host} {port}")
        return "正在检查连接"
    if a == "status":
        ok, msg = start_job("读从臂状态", SYS +
                            'timeout 4 ros2 topic echo /feedback/arm_status --once --no-daemon | grep -E "^arm_status|^err_status"; '
                            "/usr/bin/python3 ~/check_limits.py")
    elif a == "gripper":
        w = num(p, "width_mm", 0, 100) / 1000
        ok, msg = start_job(f"夹爪 → {w*1000:.0f} mm", SYS +
                            'ros2 service call /control_enable std_srvs/srv/SetBool "{data: true}" >/dev/null && '
                            f'ros2 topic pub --once /control/joint_states sensor_msgs/msg/JointState "{{name: [gripper], position: [{w:.4f}], effort: [1.0]}}" >/dev/null; '
                            "sleep 2; ~/nero_lock.sh; "
                            "echo -n '夹爪实际 (m): '; ros2 topic echo /feedback/joint_states --once --no-daemon | grep -A8 position | tail -1")
    elif a == "to_leader":
        g, sp = num(p, "gripper_mm", -1, 100), num(p, "speed", 3, 30)
        ok, msg = start_job("从臂回到主臂姿态", EVO + f"python -u ~/nero_goto_leader.py --gripper-mm {g:g} --speed-deg {sp:g}", stdin_text="yes\n")
    elif a == "to_leader_dry":
        ok, msg = start_job("从臂回到主臂姿态（只看计划）", EVO + "python -u ~/nero_goto_leader.py --dry")
    elif a == "leader_align":
        ok, msg = start_job("主臂对齐到从臂（从臂不动）", EVO + "python -u ~/nero_init.py", stdin_text="yes\n")
    elif a == "teleop":
        dur, step = num(p, "duration", 10, 600, int), num(p, "speed", 10, 90) 
        rad = round(step / 50 * 3.14159 / 180, 4)   # °/s → 每周期 rad（50 Hz）
        ok, msg = start_job(f"遥操作：拖主臂带从臂（{dur} s，≤{step:g}°/s）", SYS +
                            f"/usr/bin/python3 -u ~/teleop_tune.py {dur} {rad} 0.002 10", stdin_text="yes\n")
    elif a == "check":
        ok, msg = start_job("检查 check", EVO + f"python -u ~/nero_vla_client.py --mode check --task {task}")
    elif a == "dry":
        n = num(p, "chunks", 1, 30, int)
        sp = bool(p.get("step_prompt"))
        if sp:
            write_step(1)   # dry 按第 1 步的指令推理
        ok, msg = start_job(f"空跑 dry × {n}（{host}:{port}，步骤文本{'开' if sp else '关'}）", EVO +
                            f"python -u ~/nero_vla_client.py --mode dry --server-host {host} --server-port {port} --task {task} "
                            f"--max-chunks {n} --dry-interval-s 0.5{f' --step-file {STEP_FILE} --step-prompt' if sp else ''}")
    elif a == "replay":
        f = chunk_file(p["file"])
        k, n, s = num(p, "chunk", 0, 99, int), num(p, "steps", 1, 30, int), num(p, "slow", 1, 10)
        ok, msg = start_job(f"手动执行 第{k}块 前{n}步 ×{s:g}", EVO +
                            f"python -u ~/nero_replay_chunk.py --file {shlex.quote(f)} --chunk {k} --steps {n} --slow {s:g} --go",
                            stdin_text="yes\n")
    elif a == "live":
        e, s, n = num(p, "exec_steps", 1, 30, int), num(p, "slow", 1, 10), num(p, "chunks", 1, 100, int)
        t = num(p, "max_s", 10, 1800, int)
        auto = bool(p.get("auto_return"))
        variant = re.sub(r"[^\w.+-]", "", str(p.get("variant", "")))[:24]
        sp = bool(p.get("step_prompt"))
        check_preset(variant, host, port, sp, e, s)
        server = f"{host}:{port}"
        fan = re.sub(r"\D", "", str(p.get("fan_pos", "")))[:2]
        scene = re.sub(r"[^\w.+-]", "", str(p.get("scene", "")))[:24]
        pending_meta.clear()
        pending_meta.update(task=task, item=str(p.get("item", ""))[:40], video=str(p.get("video", ""))[:60], auto_return=auto,
                            variant=variant, step_prompt=sp, server=server, fan_pos=fan, scene=scene)
        t_start = time.time()
        trial_start(task, variant, str(p.get("item", ""))[:40], sp, server, fan, scene)

        def after(rc):   # 运行结束：先存评测；没按过急停、勾了“自动回到主臂姿态”，就 2 秒后自动执行
            trial_finish()
            if not auto:
                return
            if last_estop[0] >= t_start:
                add_log("── 运行中按过急停，不自动回到主臂姿态（确认安全后手动点）")
                return
            time.sleep(2.0)
            add_log("── 自动：从臂回到主臂姿态")
            start_job("从臂回到主臂姿态（自动）", EVO + "python -u ~/nero_goto_leader.py --gripper-mm 93 --speed-deg 15", stdin_text="yes\n")
        ok, msg = start_job(f"自动运行 live 最多 {n} 块 / {t} s", EVO +
                            f"python -u ~/nero_vla_client.py --mode live --server-host {host} --server-port {port} --task {task} "
                            f"--exec-steps {e} --slow {s:g} --max-chunks {n} --max-duration-s {t} "
                            f"--step-file {STEP_FILE} --variant {shlex.quote(variant or '-')}{' --step-prompt' if sp else ''}",
                            stdin_text="yes\n", on_done=after)
        if not ok:
            trial["active"] = False
    else:
        raise ValueError("未知操作")
    if not ok:
        raise RuntimeError(msg)
    return msg


def list_runs():
    files = sorted(glob(f"{RUNS}/*/*/chunks.npz"), reverse=True)[:30]
    out = []
    for f in files:
        try:
            n = len(np.load(f, allow_pickle=True)["latency"])
        except Exception:
            n = 0
        mode = f.split("/")[-2]
        out.append({"file": f, "label": f"{'自动运行' if mode == 'live' else '空跑'} {f.split('/')[-3]}", "chunks": n, "mode": mode})
    return out


def read_chunks(f):
    d = np.load(chunk_file(f), allow_pickle=True)
    states, acts = d["state"], d["actions"]
    res = []
    for i in range(len(acts)):
        s, a = np.asarray(states[i], float), np.asarray(acts[i], float)
        rel = np.degrees(a[:, :7] - s[:7])
        res.append({"state_deg": np.round(np.degrees(s[:7]), 1).tolist(), "state_grip_mm": round(s[7] * 1000, 1),
                    "rel_deg": np.round(rel, 1).tolist(), "grip_mm": np.round(a[:, 7] * 1000, 1).tolist(),
                    "span": round(float(np.abs(rel - rel[0]).max()), 1),
                    "latency": round(float(d["latency"][i]), 2)})
    return {"file": f, "chunks": res}


# ---------------------------------------------------------------- 人工评测
eval_lock = threading.Lock()


def task_info(task):
    """step 名字、物品叫法和名字：从 ~/nero_tasks.json 和采集时的标签文件读。"""
    t = json.loads(Path(HOME, "nero_tasks.json").read_text(encoding="utf-8")).get(task, {})
    items = {}
    try:
        items = json.loads(Path(DATA, "_labels", f"{t.get('dataset')}.json").read_text(encoding="utf-8")).get("items", {})
    except Exception:
        pass
    return {"task": task, "name": t.get("name", task), "prompt": t.get("task", ""), "steps": t.get("steps", {}),
            "item_label": t.get("item_label", "物品"), "items": items, "target": EVAL_TARGET}


def eval_path(task):
    return Path(EVALS, f"{task}.jsonl")


def load_evals(task):
    p = eval_path(task)
    if not p.exists():
        return []
    out = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out


def save_evals(task, rows):
    p = eval_path(task)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(p)


def live_runs(task):
    """这个任务的自动运行记录（新的在前），带上是否已评测。"""
    ev = {r["run"]: r for r in load_evals(task)}
    done = set(ev)
    out = []
    for f in sorted(glob(f"{RUNS}/*/live/summary.json"), reverse=True)[:200]:
        try:
            s = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue
        if s.get("args", {}).get("task") != task:
            continue
        run = f.split("/")[-3]
        meta = {}
        try:
            meta = json.loads((Path(f).parent / "meta_web.json").read_text(encoding="utf-8"))
        except Exception:
            pass
        e = ev.get(run, {})
        out.append({"run": run, "started": s.get("started") or run, "duration_s": s.get("duration_s"),
                    "item": e.get("item") or meta.get("item", ""), "video": e.get("video") or meta.get("video", ""),
                    "result": None if not e else ("成功" if e.get("success") else "失败"),
                    "chunks": s.get("executed_chunks", len(s.get("chunks", []))), "end_reason": s.get("end_reason", ""),
                    "exec_steps": s["args"].get("exec_steps"), "slow": s["args"].get("slow"), "evaluated": run in done,
                    "variant": e.get("variant") or meta.get("variant") or s.get("variant") or "",
                    "scene": e.get("scene") or meta.get("scene", ""),
                    "fail_kind": e.get("fail_kind", "")})
    for i, r in enumerate(reversed(out), 1):   # 按时间先后编号：这个任务的第几次运行
        r["no"] = i
    return out


def eval_stats(task):
    info, rows = task_info(task), load_evals(task)
    steps = list(info["steps"]) or ["1", "2", "3", "4"]
    n = len(rows)
    per = {k: {"done": sum(1 for r in rows if r["steps"].get(k) == "done"),
               "tried": sum(1 for r in rows if r["steps"].get(k) in ("done", "fail"))} for k in steps}
    return {"info": info, "rows": rows[::-1], "n": n, "success": sum(1 for r in rows if r.get("success")),
            "per_step": per, "avg_s": round(sum(r.get("duration_s") or 0 for r in rows) / n, 1) if n else 0}


def save_eval(body):
    task = task_of(body)
    run = str(body.get("run", ""))
    if not re.fullmatch(r"\d{8}_\d{6}", run) or not Path(RUNS, run, "live").is_dir():
        raise ValueError("先选一次自动运行记录")
    steps = {str(k): v for k, v in (body.get("steps") or {}).items() if v in ("done", "fail", "na")}
    if not steps:
        raise ValueError("至少标一个 step")
    meta = next((x for x in live_runs(task) if x["run"] == run), {})
    row = {"run": run, "task": task, "started": meta.get("started"), "duration_s": meta.get("duration_s"),
           "chunks": meta.get("chunks"), "exec_steps": meta.get("exec_steps"), "slow": meta.get("slow"),
           "end_reason": meta.get("end_reason"), "steps": steps, "success": bool(body.get("success")),
           "item": str(body.get("item", ""))[:40], "video": str(body.get("video", ""))[:60], "note": str(body.get("note", ""))[:300],
           "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
    old = next((r for r in load_evals(task) if r["run"] == run), {})
    for key in ("variant", "fail_kind", "step_s", "step_prompt", "server", "fan_pos", "scene"):   # 周末实验字段：⑥ 自动存，⑤ 修改时保留
        if key in body:
            row[key] = body[key]
        elif key in old:
            row[key] = old[key]
    with eval_lock:
        rows = [r for r in load_evals(task) if r["run"] != run] + [row]
        rows.sort(key=lambda r: r["run"])
        save_evals(task, rows)
    return f"已保存（{task_info(task)['name']} 共 {len(rows)} 次）"


def delete_eval(body):
    task, run = task_of(body), str(body.get("run", ""))
    with eval_lock:
        rows = load_evals(task)
        save_evals(task, [r for r in rows if r["run"] != run])
    return "已删除"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path == "/":
                self._send(200, PAGE.encode(), "text/html")
            elif u.path == "/api/state":
                with lock:
                    p = job["proc"]
                    running = p is not None and p.poll() is None
                    st = {"name": job["name"], "running": running, "rc": job["rc"],
                          "elapsed": round((time.time() if running else job["ended"] or time.time()) - job["started"], 1)
                          if job["started"] else 0}
                since = int(parse_qs(u.query).get("since", ["0"])[0])
                total = len(log)
                self._send(200, {"job": st, "log_total": total, "log": list(log)[since:] if since <= total else list(log)})
            elif u.path == "/api/runs":
                self._send(200, list_runs())
            elif u.path == "/api/eval_state":
                t = task_of({"task": parse_qs(u.query).get("task", ["weigh"])[0]})
                self._send(200, {**eval_stats(t), "runs": live_runs(t)})
            elif u.path == "/api/trial":
                self._send(200, {**trial_state(), "fail_kinds": FAIL_KINDS, "fail_common": FAIL_COMMON})
            elif u.path == "/api/plan":
                self._send(200, {"presets": load_presets(), "pings": pings, "plan": plan_state(), "eval_dir": EVALS})
            elif u.path == "/api/tasks":
                allt = load_all_tasks()
                self._send(200, {"tasks": TASKS, "report_url": REPORT,
                                 "collect": [{"key": k, "name": t.get("name", k), "dataset": t.get("dataset"), "fail": bool(t.get("failure")), "item_label": t.get("item_label", "物品")}
                                             for k, t in allt.items() if t.get("dataset") and t.get("task")]})
            elif u.path == "/api/collect_state":
                self._send(200, collect_state(parse_qs(u.query).get("preset", ["weigh"])[0]))
            elif u.path == "/api/scenes":
                k = parse_qs(u.query).get("task", ["weigh"])[0]
                S = load_scenes()
                self._send(200, {"scenes": (S.get("tasks", {}).get(k) or {}).get("scenes", [])})
            elif u.path == "/api/chunks":
                self._send(200, read_chunks(parse_qs(u.query)["file"][0]))
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:
            self._send(400, {"error": str(e)})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/eval":
                return self._send(200, {"ok": True, "msg": save_eval(body)})
            if self.path == "/api/eval_delete":
                return self._send(200, {"ok": True, "msg": delete_eval(body)})
            if self.path == "/api/trial":
                return self._send(200, {"ok": True, "msg": trial_event(str(body.get("op", "")), body)})
            if self.path == "/api/presets":
                return self._send(200, {"ok": True, "msg": save_presets(body)})
            if self.path == "/api/schedule":
                return self._send(200, {"ok": True, "msg": save_sched(body)})
            if self.path == "/api/collect":      # 采集按钮：原样转发给报告服务的 /api/ctl
                a = str(body.get("action", ""))
                j = report_call("/api/ctl", {"action": a, "preset": str(body.get("preset", ""))}, timeout=25 if a.startswith("pos_") else 6)
                return self._send(200, {"ok": True, "msg": j.get("message", "")})
            if self.path == "/api/collect_item":  # 设“当前物品”：转发给报告服务的 /api/labels
                j = report_call("/api/labels", {"dataset": str(body.get("dataset", "")), "current": int(body.get("current") or 0)})
                return self._send(200, {"ok": True, "msg": "当前物品已改，下一条开录时生效"})
            if self.path == "/api/scene":
                return self._send(200, {"ok": True, "msg": set_scene(body)})
            msg = handle_action(body.get("action", ""), body.get("params", {}))
            self._send(200, {"ok": True, "msg": msg})
        except Exception as e:
            self._send(400, {"ok": False, "msg": str(e)})


PAGE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NERO 采集与测试台</title>
<style>
:root{--bg:#f5f6f8;--card:#fff;--ink:#1d2330;--mute:#6b7385;--line:#e2e5ea;--acc:#2f6fdb;--ok:#1f9d55;--warn:#c27c0e;--bad:#d23b3b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
header{position:sticky;top:0;z-index:5;background:#fff;border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
h1{font-size:17px;margin:0 8px 0 0}.chip{padding:3px 10px;border-radius:99px;background:#eef1f5;color:var(--mute);font-size:13px}
.chip.run{background:#fff4dc;color:var(--warn)}.chip.ok{background:#e4f6ec;color:var(--ok)}.chip.err{background:#fde8e8;color:var(--bad)}
.sp{flex:1}button{font:inherit;border:1px solid var(--line);background:#fff;border-radius:8px;padding:7px 14px;cursor:pointer}
button:hover{border-color:#b9c0cc}button:disabled{opacity:.45;cursor:not-allowed}
button.pri{background:var(--acc);color:#fff;border-color:var(--acc)}button.move{background:#fff4dc;border-color:#e9c27a}
button.stopbtn{background:#eef1f5}button.estop{background:var(--bad);color:#fff;border-color:var(--bad);font-weight:700;padding:8px 18px}
main{max-width:1180px;margin:0 auto;padding:16px;display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.card h2{font-size:15px;margin:0 0 4px}.card p{margin:0 0 10px;color:var(--mute);font-size:13px}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:6px 0}
label{color:var(--mute);font-size:13px}input,select{font:inherit;border:1px solid var(--line);border-radius:6px;padding:5px 7px;width:76px;background:#fff}
select{width:auto}.wide{grid-column:1/-1}.tag{font-size:11px;padding:1px 6px;border-radius:4px;background:#fff4dc;color:var(--warn);margin-left:6px}
.tag.safe{background:#e4f6ec;color:var(--ok)}
#log{background:#111722;color:#cfd6e4;font:12px/1.45 ui-monospace,Menlo,Consolas,monospace;border-radius:8px;padding:10px;height:320px;overflow:auto;white-space:pre-wrap;margin:0}
.tbl{overflow:auto;max-height:420px;border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font:12px ui-monospace,Menlo,Consolas,monospace}th,td{padding:2px 6px;text-align:right;border-bottom:1px solid #f0f1f4}
th{position:sticky;top:0;background:#fafbfc;color:var(--mute);font-weight:500}tr.cut td{border-bottom:2px solid var(--acc)}tr.after td{opacity:.4}
.summary{font-size:13px;color:var(--mute);margin:6px 0}
.evstat{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin:6px 0 12px}
.evstat .k{background:#f6f8fb;border-radius:10px;padding:8px 10px}.evstat b{font-size:20px;display:block;font-variant-numeric:tabular-nums}.evstat span{color:var(--mute);font-size:12px}
.bar{height:6px;border-radius:9px;background:#e8ebf0;overflow:hidden;margin-top:4px}.bar i{display:block;height:100%;background:var(--ok)}
.evform{border:1px solid var(--line);border-radius:10px;padding:8px 12px;background:#fbfcfd}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}.seg button{border:0;border-radius:0;padding:5px 12px;background:#fff}
.seg button+button{border-left:1px solid var(--line)}.seg button.on[data-v="done"],.seg button.on[data-v="1"]{background:#e4f6ec;color:var(--ok);font-weight:700}
.seg button.on[data-v="fail"],.seg button.on[data-v="0"]{background:#fde8e8;color:var(--bad);font-weight:700}.seg button.on[data-v="na"]{background:#eef1f5;font-weight:700}
.stepRow{display:flex;gap:10px;align-items:center;margin:5px 0}.stepRow .nm{min-width:150px}
table.evt tr.cur td{background:#eef4ff}
table.evt{font:13px system-ui,"PingFang SC","Microsoft YaHei",sans-serif}table.evt td,table.evt th{text-align:left;padding:5px 8px}
.ok{color:var(--ok);font-weight:700}.no{color:var(--bad);font-weight:700}.na{color:var(--mute)}.flash{animation:fl 1.6s}@keyframes fl{0%{box-shadow:0 0 0 4px #f5c46a}100%{box-shadow:none}}
.trBig{margin:8px 0;padding:12px 16px;border-radius:12px;background:#eef4ff;border:2px solid var(--acc)}
.trBig .s{font-size:30px;font-weight:700}.trBig .p{font:13px ui-monospace,Menlo,Consolas,monospace;color:#333;margin-top:4px;word-break:break-all}
.presetRow button.pv{margin-right:4px}.presetRow button.pv.on{background:var(--acc);color:#fff;border-color:var(--acc)}
#presetTbl input{width:90px}#presetTbl input.w{width:180px}
dialog{border:none;border-radius:12px;padding:20px;max-width:440px}dialog::backdrop{background:#0006}
dialog h3{margin:0 0 8px}dialog ul{padding-left:18px;margin:8px 0;color:#333}
body.mode-collect .t-only{display:none!important}body:not(.mode-collect) .c-only{display:none!important}
.modeSw{display:inline-flex;border:1px solid var(--line);border-radius:9px;overflow:hidden}.modeSw button{border:0;border-radius:0;padding:6px 16px}
.modeSw button.on{background:var(--ink);color:#fff}
.cBig{margin:8px 0;padding:16px 18px;border-radius:12px;border:2px solid var(--line);background:#f6f8fb}
.cBig .s{font-size:30px;font-weight:700}.cBig .d{margin-top:4px;color:#333}
.cBig.prep,.cBig.reset{background:#fff4dc;border-color:#e9c27a}.cBig.armed{background:#e8f0ff;border-color:var(--acc)}
.cBig.rec{background:#e4f6ec;border-color:var(--ok)}.cBig.busy{background:#eef1f5}.cBig.off{background:#f6f8fb}
#cKeys button{min-width:64px}#cKeys button.k1{background:var(--acc);color:#fff;border-color:var(--acc);font-size:17px;padding:10px 22px}
#cTerm{background:#111722;color:#cfd6e4;font:12px/1.45 ui-monospace,Menlo,Consolas,monospace;border-radius:8px;padding:10px;max-height:300px;overflow:auto;white-space:pre-wrap;margin:0}
.lay{display:flex;gap:8px;flex-wrap:wrap;margin-top:6px}.lay span{background:#fff;border:1px solid var(--line);border-radius:8px;padding:4px 10px}.lay b{color:var(--acc)}
</style></head><body>
<header>
 <h1>NERO 采集与测试台</h1><span class="modeSw"><button id="mC" onclick="setMode('collect')">采集</button><button id="mT" onclick="setMode('test')">测试</button></span>
 <span id="chip" class="chip t-only">空闲</span>
 <span class="t-only"><label>任务</label><select id="task"></select>
 <label>模型服务</label><input id="host" value="172.16.11.115" style="width:118px" oninput="presetMark()"><label>:</label><input id="port" value="6699" style="width:60px" oninput="presetMark()">
 <button onclick="ping()">检查连接</button><span id="pingChip" class="chip">未检查</span></span>
 <span class="sp"></span>
 <button id="bStop" onclick="act('stop')">停止当前任务</button>
 <button onclick="act('lock')">关锁</button>
 <button class="estop" onclick="act('estop')">软件急停</button>
</header>
<main>
 <section class="card wide c-only" id="cCard"><h2>采集控制<span class="tag">录制时两台手臂都由人带着动</span></h2>
  <p>按钮和快捷键都转发给报告服务，再由它把按键送进 tmux 里的录制程序（和在终端里按一样）。快捷键：→ 开始 / 保存并开始下一条，← 丢弃，1–4 切换 step，S 成功，F 失败，Esc 结束（会先确认）。
   <b>新：</b>按 → 后先进入「就绪」，主臂一动才开始录，所以按完不用着急。</p>
  <div class="row"><label>任务预设</label><select id="cPre" onchange="cPoll()"></select>
   <button class="pri" id="cStart" onclick="cAct('start')">▶ 开始采集</button><button id="cYes" onclick="cAct('yes')">输入 yes</button>
   <span id="cMsg" class="summary"></span><span class="sp"></span><a id="cReport" target="_blank" rel="noopener">数据总览 ↗</a></div>
  <div id="cBig" class="cBig off"><div class="s">读取中…</div></div>
  <div class="row" id="cKeys"></div>
  <div class="evstat" id="cProg"></div>
 </section>
 <section class="card wide c-only" id="scCard"><h2>场景和物品<span class="tag safe">不动手臂</span></h2>
  <p>每条开录时读一次「当前场景 + 摆法」和「当前物品」，录完存进 _labels（不改数据集字段）。摆法选「自动」时按这个场景已录条数每 <b id="scEvery">5</b> 条换一次，工具轮流放到不同角标，避免模型靠位置认工具。</p>
  <div class="row"><label>场景</label><select id="scSel" onchange="scPaint()"></select><label>摆法</label><select id="scRot"><option value="auto">自动</option><option value="0">1</option><option value="1">2</option><option value="2">3</option><option value="3">4</option></select>
   <button class="pri" onclick="scSet()">设为当前场景</button><button onclick="scSet(true)">清除</button><span id="scMsg" class="summary"></span></div>
  <div id="scNow" class="trBig"></div>
  <div class="tbl"><table id="scTbl" class="evt"></table></div>
  <div class="row" style="margin-top:10px"><label id="ciLb">物品</label><select id="ciSel"></select><button onclick="ciSet()">设为当前物品</button><span id="ciMsg" class="summary"></span></div>
 </section>
 <section class="card wide c-only"><h2>录制终端<span class="tag safe">只读</span></h2><pre id="cTerm"></pre></section>
 <section class="card t-only wide"><h2>⓪ 拖主臂把从臂带到物品附近<span class="tag">两台手臂会动</span></h2>
  <p>流程：<b>1</b> 主臂对齐到从臂（主臂自己转到从臂姿态，从臂不动）→ <b>2</b> 开始遥操作，拖主臂把从臂带到物品上方约 20 cm（主从差 &gt; 10° 会自动停）→
   <b>3</b> 结束遥操作（关锁 + 原地保持）→ ① 设置夹爪 93 mm → ④ 自动运行。遥操作时从臂夹爪会跟随主臂夹爪，结束后再用 ① 张开。</p>
  <div class="row"><button class="job move" onclick="confirmRun('leader_align',{},'主臂将转到从臂当前姿态（从臂不动）。主臂旁有人、两台手臂运动范围内无障碍。')">1 主臂对齐到从臂</button>
   <label>时长</label><input id="td" value="180"><label>秒</label><label>从臂最快</label><input id="tv" value="40"><label>°/s</label>
   <button class="job move" onclick="confirmRun('teleop',{duration:v('td'),speed:v('tv')},'遥操作 '+v('td')+' 秒：拖主臂，从臂跟随（最快 '+v('tv')+'°/s）。主从差 > 3° 不会开始，> 10° 自动停。')">2 开始遥操作</button>
   <button class="stopbtn" onclick="act('stop')">3 结束遥操作</button></div>
  <div class="row" style="border-top:1px dashed var(--line);padding-top:8px;margin-top:8px"><b>每次测试前：</b>把主臂摆成起始姿态（不用开遥操作，直接用手摆），再点
   <button class="job move" onclick="confirmRun('to_leader',{gripper_mm:v('tg'),speed:v('tsp')},'从臂将平滑走到主臂现在的姿态（7 个关节同时动，最快 '+v('tsp')+'°/s），夹爪张到 '+v('tg')+' mm。末端太低时会先竖直抬高；途中会碰台面就拒绝执行。')">↩ 从臂回到主臂姿态</button>
   <label>夹爪</label><input id="tg" value="93"><label>mm</label><label>最快</label><input id="tsp" value="15"><label>°/s</label>
   <button class="job" onclick="act('to_leader_dry')">只看计划</button><span class="tag safe">只看计划不动手臂</span></div>
 </section>
 <section class="card t-only"><h2>① 准备<span class="tag safe">不动手臂</span></h2>
  <p>读状态：使能和限位。check：不连模型，检查关节、两相机能否读到。</p>
  <div class="row"><button class="job" onclick="act('status')">读从臂状态</button><button class="job" onclick="act('check')">检查 check</button></div>
  <div class="row"><label>夹爪开到</label><input id="gw" value="93"><label>mm</label>
   <button class="job move" onclick="act('gripper',{width_mm:v('gw')})">设置夹爪</button><span class="tag">只动夹爪</span></div>
 </section>
 <section class="card t-only"><h2>② 空跑 dry<span class="tag safe">不动手臂</span></h2>
  <p>连模型推理，只记录动作块不执行。手臂静止时模型常先“等” 2 秒，多采几块挑有动作的。</p>
  <div class="row"><label>采</label><input id="dn" value="8"><label>块</label><button class="job pri" onclick="act('dry',{chunks:v('dn'),step_prompt:$('lStep').checked?1:0})">开始空跑</button></div>
  <p>用顶部的模型服务；④ 勾了步骤文本就按第 1 步的指令发。日志里每块会打印「指令：…」，确认带不带 Step k/4。</p>
 </section>
 <section class="card t-only wide"><h2>③ 查看动作块 · 手动执行<span class="tag">执行时手臂会动</span></h2>
  <p>选一次 dry 记录和其中一块，看每一步相对当前姿态的变化（°），再只执行前 N 步。执行前会检查手臂与记录时相差 ≤ 2°。</p>
  <div class="row"><label>记录</label><select id="run" onchange="loadChunks()"></select><button onclick="loadRuns()">刷新</button>
   <label>块</label><select id="chunk" onchange="drawTable()"></select>
   <label>执行前</label><input id="rn" value="10" oninput="drawTable()"><label>步</label><label>放慢</label><input id="rs" value="3"><label>倍</label>
   <button class="job move" onclick="replay()">手动执行</button></div>
  <div class="summary" id="sum"></div>
  <div class="tbl"><table id="tbl"></table></div>
 </section>
 <section class="card t-only wide"><h2>④ 自动运行 live<span class="tag">手臂会动</span></h2>
  <p>模型闭环控制：推理 → 停稳 → 执行，循环。整块执行(30 步)才能跳过开头的等待。</p>
  <div class="row presetRow"><b>模型预设</b><span id="presetBtns"></span><span id="presetNow" class="summary"></span>
   <button onclick="$('presetEdit').hidden=!$('presetEdit').hidden">改预设</button></div>
  <div id="presetEdit" hidden class="evform" style="margin:6px 0"><div class="tbl"><table id="presetTbl" class="evt"></table></div>
   <div class="row"><button class="pri" onclick="presetSave()">保存预设</button><span class="summary">端口以 113 上服务实际用的为准。夹爪模式：M2 系列是 absolute，M0、M1 是 delta；检查连接时会对照。</span></div></div>
  <div class="row"><label>每块执行</label><input id="le" value="30"><label>步</label><label>放慢</label><input id="ls" value="2"><label>倍</label>
   <label>变体</label><input id="lVar" style="width:90px" placeholder="E0 / M1">
   <label style="color:var(--ink)"><input type="checkbox" id="lStep" style="width:auto"> 指令带步骤文本（M1、M2 这类模型才勾）</label></div>
  <div class="row"><label id="lItemLb">物品</label><select id="lItem"></select>
   <label>场景</label><select id="lScene"><option value="">（不填）</option></select>
   <span id="lFanBox" hidden><label>风扇位置</label><select id="lFan"><option value="">（不填）</option><option>1</option><option>2</option><option>3</option></select></span>
   <label>外接相机视频编号</label><input id="lVideo" style="width:150px" placeholder="如 VID_0012">
   <label style="color:var(--ink)"><input type="checkbox" id="lAuto" style="width:auto"> 结束后自动回到主臂姿态</label><span class="summary">（按过急停就不会自动动）</span></div>
  <div class="row"><label>最多</label><input id="ln" value="40"><label>块</label><label>最长</label><input id="lt" value="600"><label>秒</label>
   <button class="job move" onclick="liveGo()">开始自动运行</button></div>
 </section>
 <section class="card t-only wide" id="planCard"><h2>⑦ 今天的顺序和进度<span class="tag safe">不动手臂</span></h2>
  <p>先生成交替顺序（ABBA 排法，两个变体轮流）。每存一条评测，「下一次」自动往后走；标成“操作失误”的不算数，要重跑。勾选下面的框，跑完会自动套用下一次的预设。</p>
  <div class="row"><span style="font-size:22px">下一次：<b id="nextVar">—</b></span><span id="nextInfo" class="summary"></span>
   <button class="pri" id="nextApply" onclick="applyPreset(NEXT)">套用这个预设</button>
   <label style="color:var(--ink)"><input type="checkbox" id="autoNext" checked style="width:auto"> 跑完自动套用下一次的预设</label></div>
  <div class="row"><label>变体 A</label><input id="gA" value="M1" style="width:70px"><label>变体 B</label><input id="gB" value="M2" style="width:70px">
   <label>各</label><input id="gN" value="15" style="width:50px"><label>次</label><button onclick="genOrder()">生成顺序</button>
   <label>顺序（逗号分隔，可手改）</label><input id="gOrder" style="width:min(100%,420px)"><button onclick="saveOrder()">保存顺序</button><span id="planMsg" class="summary"></span></div>
  <div class="tbl"><table id="planTbl" class="evt"></table></div>
  <p class="summary" id="planDir"></p>
 </section>
 <section class="card t-only wide" id="trialCard"><h2>⑥ 边跑边标：本次运行<span class="tag safe">按钮不动手臂</span></h2>
  <p>④ 开始自动运行后在这里操作。<b>本步完成</b>的时刻和采集时按键一样：上一步结束就按（测水温第 3 步在提起探针后按）。
   一失败就按对应的失败按钮，会停止运行并自动存评测；4 步都完成也会自动停止并记为成功。快捷键：N 本步完成，1–4 失败，Z 撤销。</p>
  <div id="trTop" class="summary">还没有开始</div>
  <div id="trBig" class="trBig" hidden></div>
  <div class="evstat" id="trSteps"></div>
  <div class="row"><button class="pri" id="trDone" style="font-size:18px;padding:12px 28px" onclick="tr('step_done')">本步完成 (N)</button>
   <button onclick="tr('undo')">撤销 (Z)</button>
   <label>失败类型</label><select id="trKind"></select>
   <button class="move" onclick="trFail()">本步失败 (1–4)</button></div>
  <div class="row"><label>备注</label><input id="trNote" style="width:min(100%,520px)" placeholder="比如：夹偏了 / 放在秤边缘"><button onclick="tr('note',{note:v('trNote')})">记下</button><span id="trMsg" class="summary"></span></div>
  <div class="evstat" id="trLive"></div>
 </section>
 <section class="card t-only wide" id="evalCard"><h2>⑤ 评测：每次自动运行后人工打勾<span class="tag safe">不动手臂</span></h2>
  <p>每个任务做 <b id="evTarget">20</b> 次。下面的表是这个任务的全部自动运行，按时间编号“第 N 次”，开跑前在 ④ 填的物品和视频编号会跟着这一次。点一行的“评测”，按外接相机的录像判断每个 step 是否完成，保存。前一个 step 没完成，后面的标“没到这步”。</p>
  <div class="evstat" id="evStat"></div>
  <div class="evform">
   <div class="row"><label>正在评测</label><select id="evRun" onchange="evPick()"></select><span id="evRunInfo" class="summary"></span></div>
   <div class="row"><label id="evItemLb">物品</label><select id="evItem"></select><label>外接相机视频编号</label><input id="evVideo" style="width:150px" placeholder="如 VID_0012"></div>
   <div id="evSteps"></div>
   <div class="row"><label>整体</label><span class="seg" id="evSucc"><button data-v="1">✓ 成功</button><button data-v="0">✗ 失败</button></span>
    <span class="summary">（4 个 step 都完成会自动选“成功”，也可以手动改）</span></div>
   <div class="row"><label>备注</label><input id="evNote" style="width:min(100%,560px)" placeholder="比如：夹偏了 / 放在秤边缘 / 在物体上方停住不动"></div>
   <div class="row"><button class="pri" onclick="evSave()">保存评测</button><button onclick="evReset()">清空</button><span id="evMsg" class="summary"></span></div>
  </div>
  <h3 style="font-size:14px;margin:14px 0 6px">运行记录 <span class="summary" id="evCount"></span></h3>
  <div class="tbl" style="max-height:420px"><table id="evTbl" class="evt"></table></div>
 </section>
 <section class="card t-only wide"><h2>日志</h2><pre id="log"></pre></section>
</main>
<dialog id="dlg"><h3>确认：手臂将运动</h3><div id="dlgText" style="white-space:pre-line"></div>
 <ul><li>手放实体急停</li><li>从臂周围已清空，离台面 / 物体 ≥ 10 cm</li><li>采集录制程序没在运行，定位网页没在 Move</li></ul>
 <label><input type="checkbox" id="dlgOk" style="width:auto"> 以上都已确认</label>
 <div class="row" style="justify-content:flex-end;margin-top:12px"><button onclick="dlg.close()">取消</button><button class="move" id="dlgGo">执行</button></div></dialog>
<script>
const $=id=>document.getElementById(id), v=id=>$(id).value.trim();
let since=0, chunks=null, running=false;
async function post(action,params={}){params.task=v('task');params.port=v('port');params.host=v('host');
 const r=await fetch('/api/run',{method:'POST',body:JSON.stringify({action,params})});const j=await r.json();
 if(!j.ok)alert(j.msg);return j}
function act(a,p){return post(a,p)}
function confirmRun(a,p,text){$('dlgText').textContent=text;$('dlgOk').checked=false;
 $('dlgGo').onclick=()=>{if(!$('dlgOk').checked){alert('请先勾选确认');return}dlg.close();post(a,p)};dlg.showModal()}
function replay(){const c=cur();if(!c){alert('先选一块');return}
 const n=+v('rn'), k=+v('chunk'); const mx=Math.max(...c.rel_deg[Math.min(n,30)-1].map(Math.abs));
 confirmRun('replay',{file:v('run'),chunk:k,steps:n,slow:v('rs')},`第 ${k} 块，执行前 ${n} 步（放慢 ${v('rs')} 倍），最后一步相对现在最大 ${mx.toFixed(1)}°`)}
async function poll(){try{const r=await fetch('/api/state?since='+since);const j=await r.json();
 if(j.log_total<since){since=0;$('log').textContent=''}
 if(j.log.length){const L=$('log'),atEnd=L.scrollTop+L.clientHeight>=L.scrollHeight-20;L.textContent+=j.log.join('\n')+'\n';if(atEnd)L.scrollTop=L.scrollHeight}
 since=j.log_total;const s=j.job,c=$('chip');running=s.running;
 if(s.running){c.className='chip run';c.textContent='运行中：'+s.name+' · '+s.elapsed+' s'}
 else if(s.name){c.className='chip '+(s.rc===0?'ok':'err');c.textContent=(s.rc===0?'完成：':'结束(返回码 '+s.rc+')：')+s.name;
  if(wasRunning&&/dry/.test(s.name))loadRuns(true);
  if(wasRunning&&/自动运行/.test(s.name)){setTimeout(()=>loadPlan(true),1500);if(v('lVideo'))$('lVideo').value=bumpVideo(v('lVideo'));evLoad(true).then(()=>{const c=$('evalCard');c.classList.remove('flash');void c.offsetWidth;c.classList.add('flash');c.scrollIntoView({behavior:'smooth'})})}}
 else{c.className='chip';c.textContent='空闲'}
 document.querySelectorAll('button.job').forEach(b=>b.disabled=s.running);$('bStop').disabled=!s.running;wasRunning=s.running}catch(e){}}
let wasRunning=false;
async function loadRuns(newest=false){const r=await (await fetch('/api/runs')).json();const sel=$('run'),old=sel.value;
 sel.innerHTML=r.map(x=>`<option value="${x.file}">${x.label} · ${x.chunks} 块</option>`).join('')||'<option value="">（还没有 dry 记录）</option>';
 if(r.length)sel.value=!newest&&old&&r.some(x=>x.file===old)?old:r[0].file;loadChunks()}
async function loadChunks(){chunks=null;$('chunk').innerHTML='';$('tbl').innerHTML='';$('sum').textContent='';if(!v('run'))return;
 const j=await (await fetch('/api/chunks?file='+encodeURIComponent(v('run')))).json();if(j.error){$('sum').textContent=j.error;return}
 chunks=j.chunks;let best=0;chunks.forEach((c,i)=>{if(c.span>chunks[best].span)best=i});
 $('chunk').innerHTML=chunks.map((c,i)=>`<option value="${i}">第 ${i} 块 · 最大变化 ${c.span}°</option>`).join('');$('chunk').value=best;drawTable()}
const cur=()=>chunks&&chunks[+v('chunk')];
function heat(x){const a=Math.min(Math.abs(x)/15,1);return a<0.03?'':`background:rgba(${x>0?'47,111,219':'210,59,59'},${(a*0.45).toFixed(2)})`}
function drawTable(){const c=cur();if(!c)return;const n=Math.max(1,Math.min(30,+v('rn')||1));
 $('sum').innerHTML=`记录时姿态(°) ${c.state_deg.join(', ')} · 夹爪 ${c.state_grip_mm} mm · 推理 ${c.latency} s · 蓝 = 增大，红 = 减小`;
 let h='<tr><th>步</th>'+[1,2,3,4,5,6,7].map(i=>'<th>j'+i+'</th>').join('')+'<th>夹爪 mm</th></tr>';
 c.rel_deg.forEach((r,i)=>{h+=`<tr class="${i===n-1?'cut':i>=n?'after':''}"><td>${i}</td>`+r.map(x=>`<td style="${heat(x)}">${x.toFixed(1)}</td>`).join('')+`<td>${c.grip_mm[i].toFixed(1)}</td></tr>`});
 $('tbl').innerHTML=h}
/* ---------- ⑤ 评测 ---------- */
let EV=null, evSel={}, evSucc=null, evAuto=true;
async function evLoad(pickNewest=false){const j=await (await fetch('/api/eval_state?task='+v('task'))).json();EV=j;
 const I=j.info, steps=Object.entries(I.steps), per=j.per_step;$('evTarget').textContent=I.target;$('evItemLb').textContent=I.item_label;
 $('evStat').innerHTML=`<div class="k"><b>${j.n} / ${I.target}</b><span>${esc(I.name)} 已评测</span><div class="bar"><i style="width:${Math.min(100,j.n/I.target*100)}%"></i></div></div>`+
  `<div class="k"><b>${j.n?Math.round(j.success/j.n*100):0}%</b><span>整体成功（${j.success} / ${j.n}）</span></div>`+
  steps.map(([k,nm])=>{const p=per[k]||{done:0,tried:0};return `<div class="k"><b>${p.tried?Math.round(p.done/p.tried*100):0}%</b><span>${k} ${esc(nm)}（${p.done} / ${p.tried}）</span><div class="bar"><i style="width:${p.tried?p.done/p.tried*100:0}%"></i></div></div>`}).join('')+
  `<div class="k"><b>${j.avg_s} s</b><span>平均每次用时</span></div>`;
 const old=$('evRun').value, runs=j.runs;
 $('evRun').innerHTML=runs.map(r=>`<option value="${r.run}">第 ${r.no} 次 · ${esc(String(r.started).slice(5,16))}${r.item?' · '+esc(r.item):''}${r.video?' · '+esc(r.video):''} · ${r.evaluated?'已评测 '+r.result:'未评测'}</option>`).join('')||'<option value="">（这个任务还没有自动运行记录）</option>';
 const lsel=$('lItem'),lold=lsel.value;lsel.innerHTML='<option value="">（不填）</option>'+Object.entries(I.items).map(([k,n])=>`<option value="${k} ${esc(n||'')}">${k} · ${esc(n||'未命名')}</option>`).join('');lsel.value=lold;$('lItemLb').textContent=I.item_label;
 const firstNew=runs.find(r=>!r.evaluated);
 $('evRun').value=pickNewest&&runs.length?runs[0].run:(old&&runs.some(r=>r.run===old)?old:(firstNew||runs[0]||{}).run||'');
 $('evItem').innerHTML='<option value="">（不填）</option>'+Object.entries(I.items).map(([k,n])=>`<option value="${k} ${esc(n||'')}">${k} · ${esc(n||'未命名')}</option>`).join('');
 $('evSteps').innerHTML=steps.map(([k,nm])=>`<div class="stepRow"><span class="nm"><b>${k}</b> ${esc(nm)}</span><span class="seg" data-k="${k}"><button data-v="done">✓ 完成</button><button data-v="fail">✗ 未完成</button><button data-v="na">– 没到这步</button></span></div>`).join('');
 document.querySelectorAll('#evSteps .seg button').forEach(b=>b.onclick=()=>evMark(b.parentElement.dataset.k,b.dataset.v));
 document.querySelectorAll('#evSucc button').forEach(b=>b.onclick=()=>{evSucc=b.dataset.v==='1';evAuto=false;evPaint()});
 const byRun=Object.fromEntries(j.rows.map(r=>[r.run,r]));
 $('evCount').textContent=`共 ${runs.length} 次运行，已评测 ${runs.filter(r=>r.evaluated).length} 次`;
 $('evTbl').innerHTML='<tr><th>次</th><th>开始</th><th>变体</th><th>'+esc(I.item_label)+'</th><th>视频</th><th>块 / 用时</th><th>结束原因</th>'+steps.map(([k,nm])=>`<th title="${esc(nm)}">${k}</th>`).join('')+'<th>整体</th><th>备注</th><th></th></tr>'+
  runs.map(r=>{const e=byRun[r.run];return `<tr class="${r.run===v('evRun')?'cur':''}"><td><b>${r.no}</b></td><td>${esc(String(r.started).slice(5,16))}</td><td><b>${esc(r.variant||'')}</b></td><td>${esc(r.item||'')}</td><td>${esc(r.video||'')}</td><td>${r.chunks} 块 · ${r.duration_s ?? '?'} s</td><td class="na">${esc(r.end_reason||'')}</td>`+
  steps.map(([k])=>{const x=e&&e.steps[k];return `<td class="${x==='done'?'ok':x==='fail'?'no':'na'}">${x==='done'?'✓':x==='fail'?'✗':x==='na'?'–':''}</td>`}).join('')+
  `<td class="${e?(e.success?'ok':'no'):'na'}">${e?(e.success?'成功':'失败'):'未评测'}</td><td>${esc(e?.note||'')}</td><td style="white-space:nowrap"><button onclick="evEdit('${r.run}')">${e?'改':'评测'}</button>${e?` <button onclick="evDel('${r.run}')">删</button>`:''}</td></tr>`}).join('');
 evPick()}
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function evPick(){const r=(EV?.runs||[]).find(x=>x.run===v('evRun'));$('evRunInfo').textContent=r?`每块 ${r.exec_steps} 步 · 放慢 ${r.slow} 倍 · 结束：${r.end_reason||'—'}`:'';
 document.querySelectorAll('#evTbl tr').forEach(tr=>tr.classList.remove('cur'));
 const old=(EV?.rows||[]).find(x=>x.run===v('evRun'));if(old)evFill(old);else{evReset(true);if(r){if(r.item)$('evItem').value=r.item;if(r.video)$('evVideo').value=r.video}}}
function evFill(r){evSel={...r.steps};evSucc=r.success;evAuto=false;$('evItem').value=r.item||'';$('evVideo').value=r.video||'';$('evNote').value=r.note||'';evPaint()}
function evEdit(run){$('evRun').value=run;evPick();$('evalCard').scrollIntoView({behavior:'smooth'})}
function evReset(keepRun){evSel={};evSucc=null;evAuto=true;$('evNote').value='';$('evMsg').textContent='';
 // 视频编号：在最近保存的那条上自动加 1（VID_0012 → VID_0013），物品保持不变（通常连续测同一个）
 const last=(EV?.rows||[]).slice().sort((a,b)=>(b.saved||'').localeCompare(a.saved||''))[0], m=(last?.video||'').match(/^(.*?)(\d+)(\D*)$/);
 $('evVideo').value=m?m[1]+String(+m[2]+1).padStart(m[2].length,'0')+m[3]:'';evPaint()}
function evMark(k,val){evSel[k]=val;const ks=Object.keys(EV.info.steps);
 if(val==='fail'||val==='na')ks.filter(x=>+x>+k).forEach(x=>{if(!evSel[x]||evSel[x]==='done')evSel[x]='na'});
 if(evAuto&&ks.every(x=>evSel[x]))evSucc=ks.every(x=>evSel[x]==='done');evPaint()}
function evPaint(){document.querySelectorAll('#evSteps .seg').forEach(s=>s.querySelectorAll('button').forEach(b=>b.classList.toggle('on',evSel[s.dataset.k]===b.dataset.v)));
 document.querySelectorAll('#evSucc button').forEach(b=>b.classList.toggle('on',evSucc!==null&&(b.dataset.v==='1')===evSucc))}
function bumpVideo(x){const m=(x||'').match(/^(.*?)(\d+)(\D*)$/);return m?m[1]+String(+m[2]+1).padStart(m[2].length,'0')+m[3]:''}
async function evSave(){if(!v('evRun')){alert('还没有自动运行记录');return}const ks=Object.keys(EV.info.steps);
 if(!ks.every(k=>evSel[k])){alert('每个 step 都要选：完成 / 未完成 / 没到这步');return}if(evSucc===null){alert('选一下整体成功还是失败');return}
 const r=await (await fetch('/api/eval',{method:'POST',body:JSON.stringify({task:v('task'),run:v('evRun'),steps:evSel,success:evSucc,item:v('evItem'),video:v('evVideo'),note:v('evNote')})})).json();
 $('evMsg').textContent=r.msg;if(r.ok){await evLoad();const nx=EV.runs.find(x=>!x.evaluated);if(nx){$('evRun').value=nx.run;evPick()}}}
async function evDel(run){if(!confirm('删除这条评测？（运行记录本身不删）'))return;await fetch('/api/eval_delete',{method:'POST',body:JSON.stringify({task:v('task'),run})});evLoad()}
$('task').addEventListener('change',()=>{evLoad();loadPlan()});
/* ---------- 预设 / 连接检查 / ⑦ 顺序和进度 ---------- */
let PL=null, NEXT=null;
let TASKN={weigh:'称重',task2:'水温测量',task3:'风速测量'};
const preset=name=>(PL?.presets||[]).find(p=>p.name===name);
async function loadPlan(afterRun=false){try{PL=await (await fetch('/api/plan')).json()}catch(e){return}
 $('presetBtns').innerHTML=PL.presets.map(p=>`<button class="pv" data-n="${esc(p.name)}" title="${esc(p.note||'')}" onclick="applyPreset(this.dataset.n)">${esc(p.name)}</button>`).join('');
 if($('presetEdit').hidden)$('presetTbl').innerHTML='<tr><th>名字</th><th>服务地址</th><th>端口</th><th>步骤文本</th><th>每块执行</th><th>放慢</th><th>夹爪模式</th><th>说明</th></tr>'+
  PL.presets.map((p,i)=>`<tr data-i="${i}"><td><input data-k="name" value="${esc(p.name)}"></td><td><input data-k="host" value="${esc(p.host)}" class="w"></td><td><input data-k="port" value="${p.port}"></td>`+
  `<td><input data-k="step_prompt" type="checkbox" style="width:auto" ${p.step_prompt?'checked':''}></td><td><input data-k="exec_steps" value="${p.exec_steps}"></td><td><input data-k="slow" value="${p.slow}"></td>`+
  `<td><select data-k="gripper_mode"><option ${p.gripper_mode!=='absolute'?'selected':''}>delta</option><option ${p.gripper_mode==='absolute'?'selected':''}>absolute</option></select></td><td><input data-k="note" value="${esc(p.note||'')}" class="w"></td></tr>`).join('');
 const T=PL.plan[v('task')], sc=T.schedule, nx=T.next;
 NEXT=nx&&nx.variant;$('nextVar').textContent=NEXT||(sc?'全部跑完':'（还没排顺序）');
 $('nextInfo').textContent=sc?`${TASKN[v('task')]} 第 ${Math.min(nx.done+1,nx.total)} / ${nx.total} 次`+(NEXT&&!preset(NEXT)?' · 没有同名预设，手动设置':''):'';
 $('nextApply').disabled=!NEXT||!preset(NEXT);
 if(document.activeElement!==$('gOrder'))$('gOrder').value=sc?sc.order.join(','):'';
 const vs=new Set();Object.values(PL.plan).forEach(t=>Object.keys(t.per_variant).forEach(x=>vs.add(x)));
 const tasks=Object.keys(PL.plan);
 $('planTbl').innerHTML='<tr><th>变体</th>'+tasks.map(t=>`<th>${TASKN[t]}：次数 · 成功 · 第1步 · 各步完成</th>`).join('')+'</tr>'+
  [...vs].sort().map(x=>'<tr><td><b>'+esc(x)+'</b></td>'+tasks.map(t=>{const d=PL.plan[t].per_variant[x];if(!d)return '<td class="na">—</td>';
   const tg=d.target?` / ${d.target}`:'';return `<td>${d.n}${tg} 次 · <span class="ok">${d.success}</span> · ${d.n?Math.round(d.s1/d.n*100):0}% · <span class="na">${d.reach.join('/')}</span></td>`}).join('')+'</tr>').join('');
 $('planDir').textContent='评测记录目录：'+PL.eval_dir+'（每天一个目录；“操作失误”不计入次数）';
 if(afterRun&&$('autoNext').checked&&NEXT&&preset(NEXT)){applyPreset(NEXT);const c=$('planCard');c.classList.remove('flash');void c.offsetWidth;c.classList.add('flash')}
 presetMark()}
function applyPreset(name){const p=preset(name);if(!p){alert('没有预设 '+name);return}
 $('host').value=p.host;$('port').value=p.port;$('lVar').value=p.name;$('lStep').checked=!!p.step_prompt;$('le').value=p.exec_steps;$('ls').value=p.slow;presetMark()}
function presetMark(){if(!PL)return;const p=preset(v('lVar'));
 document.querySelectorAll('#presetBtns .pv').forEach(b=>b.classList.toggle('on',b.dataset.n===v('lVar')));
 let txt='';if(p){const bad=[];if(p.host!==v('host')||String(p.port)!==v('port'))bad.push('服务');if(!!p.step_prompt!==$('lStep').checked)bad.push('步骤文本');
  if(String(p.exec_steps)!==v('le'))bad.push('每块执行');if(+p.slow!==+v('ls'))bad.push('放慢');
  txt=bad.length?`⚠ 变体 ${p.name} 的 ${bad.join('、')} 和预设不一致，开跑会被拒绝`:`当前：${p.name} → ${p.host}:${p.port} · 步骤文本${p.step_prompt?'开':'关'} · 每块 ${p.exec_steps} 步 · ${p.note||''}`}
 else txt=v('lVar')?`变体 ${v('lVar')} 不是预设，设置不做检查`:'没选预设';
 $('presetNow').textContent=txt;pingPaint()}
function pingPaint(){const key=v('host')+':'+v('port'), r=PL?.pings?.[key], c=$('pingChip');
 if(!r){c.className='chip';c.textContent='未检查';return}
 const age=Math.round((Date.now()/1000-r.t)/60), p=preset(v('lVar'));
 if(!r.ok){c.className='chip err';c.textContent=`连不上（${age} 分钟前）`;c.title=r.err||'';return}
 const gm=(r.md&&r.md.gripper_mode)||'delta', want=p&&p.gripper_mode, bad=want&&want!==gm;
 c.className='chip '+(bad?'err':'ok');c.textContent=`${bad?'⚠ 夹爪模式 '+gm+'，'+p.name+' 应为 '+want:'连接正常 · 夹爪 '+gm} · ${age} 分钟前`;c.title=JSON.stringify(r.md)}
async function ping(){await post('ping');setTimeout(loadPlan,2500);setTimeout(loadPlan,6000)}
async function presetSave(){const rows=[...document.querySelectorAll('#presetTbl tr[data-i]')].map(tr=>{const o={};
  tr.querySelectorAll('[data-k]').forEach(el=>o[el.dataset.k]=el.type==='checkbox'?el.checked:el.value.trim());return o});
 const r=await (await fetch('/api/presets',{method:'POST',body:JSON.stringify({presets:rows})})).json();alert(r.msg);loadPlan()}
function genOrder(){const A=v('gA'),B=v('gB'),n=+v('gN');if(!A||!B||!(n>0)){alert('填好变体 A、B 和次数');return}
 const o=[];for(let i=0;i<n;i++)o.push(...(i%2?[B,A]:[A,B]));$('gOrder').value=o.join(',')}
async function saveOrder(){const order=v('gOrder').split(/[,，\s]+/).filter(Boolean);
 if(PL?.plan[v('task')]?.schedule&&!confirm('覆盖这个任务原来的顺序？（“下一次”会从这一刻重新数）'))return;
 const r=await (await fetch('/api/schedule',{method:'POST',body:JSON.stringify({task:v('task'),order})})).json();$('planMsg').textContent=r.msg;loadPlan()}
function liveGo(){const p=preset(v('lVar')), key=v('host')+':'+v('port'), pg=PL?.pings?.[key];
 const pingTxt=!pg?'⚠ 这个服务还没检查连接':!pg.ok?'⚠ 上次检查连接失败':`连接 ${Math.round((Date.now()/1000-pg.t)/60)} 分钟前正常，夹爪 ${(pg.md&&pg.md.gripper_mode)||'delta'}`;
 confirmRun('live',{exec_steps:v('le'),slow:v('ls'),chunks:v('ln'),max_s:v('lt'),item:v('lItem'),video:v('lVideo'),fan_pos:v('task')==='task3'?v('lFan'):'',scene:v('lScene'),
  auto_return:$('lAuto').checked?1:0,variant:v('lVar'),step_prompt:$('lStep').checked?1:0},
  `变体 ${v('lVar')||'（没填）'}${p?'（预设）':''} → ${key}，步骤文本${$('lStep').checked?'开':'关'}。${pingTxt}。\n自动运行：每块执行 ${v('le')} 步、放慢 ${v('ls')} 倍，最多 ${v('ln')} 块 / ${v('lt')} 秒。运行中随时可按顶部“软件急停”或“停止当前任务”。`)}
function fanBox(){$('lFanBox').hidden=v('task')!=='task3'}
$('task').addEventListener('change',()=>{fanBox();loadScenesFor()});
['lVar','le','ls'].forEach(id=>$(id).addEventListener('input',presetMark));$('lStep').addEventListener('change',presetMark);
/* ---------- ⑥ 边跑边标 ---------- */
let TR=null, trKindsStep=0;
const STEPNAMES=()=>Object.values(EV?.info?.steps||{1:'1',2:'2',3:'3',4:'4'});
async function tr(op,extra={}){const r=await (await fetch('/api/trial',{method:'POST',body:JSON.stringify({op,...extra})})).json();
 $('trMsg').textContent=r.msg;trPoll()}
function trFail(){const k=v('trKind');if(!k){alert('先选失败类型');return}tr('fail',{kind:k})}
function tone(ratio){return ratio==null?'':ratio>3?'background:#fde8e8':ratio>2?'background:#fff4dc':''}
async function trPoll(){try{const s=await (await fetch('/api/trial')).json();TR=s;
 if(!s.task){$('trTop').textContent='还没有开始';return}
 const names=s.step_names||STEPNAMES(), cur=s.step;
 $('trTop').innerHTML=`变体 <b>${esc(s.variant||'-')}</b> · ${esc(s.task_name||s.task)} · ${esc(s.item||'')} · 步骤文本${s.step_prompt?'开':'关'} · `+
  (s.active?`<b>进行中</b> 已用 ${s.total_elapsed} s`:(s.result==='success'?'<span class="ok">成功</span>':s.result==='fail'?`<span class="no">第 ${cur} 步失败：${esc(s.fail_kind||'')}</span>`:'已结束（未标结果）'))+
  (s.saved?' · 评测已保存':'')+(s.server?' · 服务 '+esc(s.server):'')+(s.fan_pos?' · 风扇位置 '+esc(s.fan_pos):'')+(s.scene?' · 场景 '+esc(s.scene):'');
 const nm=(s.step_names||STEPNAMES())[cur-1]||'', Lv=s.live;$('trBig').hidden=false;
 $('trBig').innerHTML=s.active?`<div class="s">第 ${cur} 步 · ${esc(nm)}</div><div class="p">模型最近一次收到：${esc(Lv?.prompt_used||'（还没推理）')}${Lv&&s.step_prompt&&+Lv.step!==cur?' <b style="color:var(--warn)">← 新步骤下一块才生效</b>':''}</div>`
  :`<div class="s" style="font-size:20px">${s.result==='success'?'本次成功':s.result==='fail'?'第 '+cur+' 步失败':'已结束'}</div>`;
 const done={};(s.events||[]).forEach(e=>{done[e.step]=e});
 $('trSteps').innerHTML=[1,2,3,4].map(k=>{const e=done[k],isCur=k===cur&&s.active;
  const ratio=isCur&&s.demo_step_s?s.step_elapsed/s.demo_step_s:null;
  const txt=e?(e.op==='done'?`✓ ${e.dur} s`:`✗ ${esc(e.kind||'')}`):isCur?`${s.step_elapsed} s（示范约 ${s.demo_step_s??'?'} s）`:'';
  return `<div class="k" style="${isCur?'outline:2px solid var(--acc);':''}${tone(ratio)}"><b>${k}</b><span>${esc(names[k-1]||'')}</span><div>${txt}</div></div>`}).join('');
 if(trKindsStep!==cur){trKindsStep=cur;$('trKind').innerHTML='<option value="">（选一个）</option>'+[...(s.fail_kinds[cur]||[]),...s.fail_common].map(x=>`<option>${esc(x)}</option>`).join('')}
 $('trDone').disabled=!s.active;
 const L=s.live;$('trLive').innerHTML=!L?'':[
  ['第 '+L.chunk+' 块','当前块'],[L.step||'-','模型收到的步骤'],[L.latency+' s','推理耗时'],
  [L.move_deg+'°','本块计划移动（< 1° 是停滞）',L.move_deg<1?'background:#fff4dc':''],
  [L.age_ms.join(' / ')+' ms','头部 / 腕部画面年龄'],[L.sync_ms+' ms','两相机时间差',L.sync_ms>50?'background:#fff4dc':''],
  [L.grip_now_mm+' mm','夹爪实际开度'],[L.grip_pred_mm.join(' ~ ')+' mm','模型给的夹爪范围'],
  [L.grip_clipped_ticks,'夹爪被裁剪的控制周期数',L.grip_clipped_ticks>0?'background:#fff4dc':''],
 ].map(([b,sp,st])=>`<div class="k" style="${st||''}"><b style="font-size:16px">${esc(b)}</b><span>${esc(sp)}</span></div>`).join('')+
 `<div class="k" style="grid-column:1/-1"><span>模型实际收到的指令</span><div style="font:12px ui-monospace,monospace">${esc(L.prompt_used)}</div></div>`;
}catch(e){}}
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName))return;
 if(MODE==='collect'){cKey(e);return}
 if(!TR||!TR.active)return;const k=e.key.toLowerCase();
 if(k==='n'){e.preventDefault();tr('step_done')}else if(k==='z'){e.preventDefault();tr('undo')}
 else if('1234'.includes(k)&&k){e.preventDefault();$('trKind').focus()}});
setInterval(trPoll,700);
/* ---------- 11-09：采集模式（转发给报告服务）和场景 ---------- */
let MODE='test', CS=null, CBUSY=false, lastMode=null, COLL=[];
const CALLOW={prep:['next','esc'],armed:['next','esc'],start:['esc'],rec:['1','2','3','4','5','6','s','f','discard','esc'],reset:['next','discard','esc'],busy:[]};
const CMODE={prep:'准备（不录）',armed:'就绪 · 动主臂就开始录',start:'启动中',rec:'● 录制中',reset:'复位（不录）',busy:'保存中…'};
function setMode(m){MODE=m;document.body.classList.toggle('mode-collect',m==='collect');$('mC').classList.toggle('on',m==='collect');$('mT').classList.toggle('on',m!=='collect');
 try{localStorage.setItem('nero_mode',m)}catch(e){}if(m==='collect')cPoll()}
async function loadTasks(){try{const j=await (await fetch('/api/tasks')).json();TASKN=j.tasks;COLL=j.collect;
 const t=$('task'),old=t.value;t.innerHTML=Object.entries(j.tasks).map(([k,n])=>`<option value="${esc(k)}">${esc(n)}</option>`).join('');if(old&&j.tasks[old])t.value=old;
 let pold=null;try{pold=localStorage.getItem('nero_cpre')}catch(e){}
 $('cPre').innerHTML=j.collect.map(c=>`<option value="${esc(c.key)}">${esc(c.name)} → ${esc(c.dataset)}</option>`).join('');if(pold&&j.collect.some(c=>c.key===pold))$('cPre').value=pold;
 $('cReport').href=j.report_url+'/';}catch(e){}}
function beep(){try{const a=new (window.AudioContext||window.webkitAudioContext)(),o=a.createOscillator(),g=a.createGain();o.frequency.value=880;g.gain.value=0.12;o.connect(g);g.connect(a.destination);o.start();setTimeout(()=>{o.stop();a.close()},180)}catch(e){}}
async function cPoll(){if(MODE!=='collect')return;try{const j=await (await fetch('/api/collect_state?preset='+encodeURIComponent(v('cPre')))).json();CS=j;cPaint()}catch(e){}}
function cMode(){const st=CS?.ctl,lv=st?.live;return !st?'err':!st.running?'off':(lv&&lv.fresh?lv.mode:'start')}
function cPaint(){const st=CS?.ctl,lv=st?.live,m=cMode(),box=$('cBig');
 if(CS?.preset&&v('cPre')!==CS.preset&&lv?.fresh)$('cPre').value=CS.preset;
 try{localStorage.setItem('nero_cpre',v('cPre'))}catch(e){}
 if(m==='rec'&&lastMode&&lastMode!=='rec')beep();lastMode=m;
 box.className='cBig '+(['prep','armed','rec','reset','busy'].includes(m)?m:'off');
 let s='',d='';
 if(m==='err'){s='连不上报告服务';d=esc(CS?.ctl_error||'')}
 else if(m==='off'){s='录制程序未运行';d=st.pane_busy?'终端里有程序在运行（开录前检查要十几秒；要求输入 yes 时点「输入 yes」）':'选好任务预设，点 ▶ 开始采集'}
 else if(m==='start'){s='启动中…';d='开录前检查：从臂、相机、主从对齐'}
 else{const steps=lv.steps||{};s=`${CMODE[m]||m} · 第 ${lv.ep} 条`;
  if(m==='rec')s+=` · step ${lv.step} ${esc(lv.step_name||'')}`;
  if(m==='reset')s+=` · 已标「${lv.label==='success'?'成功':lv.label==='failure'?'失败':'未标'}」`;
  d=`${esc(lv.item_label||'物品')}：${esc(lv.item_name||'未选')} · 场景：${lv.scene?esc(lv.scene.scene)+' 摆法 '+(lv.scene.rot+1)+(lv.scene.desc?'（'+esc(lv.scene.desc)+'）':''):'未选'}`+
   (m==='rec'?'<br>'+Object.entries(steps).map(([k,n])=>+k===lv.step?`<b>[${k} ${esc(n)}]</b>`:+k<lv.step?`${k} ${esc(n)} ✓`:`${k} ${esc(n)}`).join('　'):'')+
   (m==='prep'?'<br>摆好场景、定位好，按 → 进入就绪':m==='armed'?'<br>手放主臂上，一动就开始录；不想等就按 →':m==='reset'?'<br>复位、定位好下一条，按 → 保存这条并进入下一条的就绪；标错了按 ← 丢弃':'')+
   (lv.positioning?'<br><b style="color:var(--warn)">定位模式开着</b>':'')}
 box.innerHTML=`<div class="s">${s}</div><div class="d">${d}</div>`;
 const ok=k=>(CALLOW[m]||[]).includes(k)&&!CBUSY, fail=!!lv?.fail;
 const B=(a,label,cls='',dis=false)=>`<button class="${cls}" ${dis?'disabled':''} onclick="cAct('${a}')">${label}</button>`;
 const steps=Object.keys(lv?.steps||{1:1,2:2,3:3,4:4});
 $('cKeys').innerHTML=B('pos_on','定位开','move',!st?.running||m==='rec'||m==='armed'||lv?.positioning||CBUSY)+B('pos_off','定位关','',!lv?.positioning||CBUSY)+
  B('next',m==='reset'?`→ 保存第 ${lv.ep} 条，进入下一条`:m==='armed'?'→ 立刻开始录':m==='rec'?'→（录制中用 S / F 结束）':'→ 进入就绪','k1',!ok('next'))+B('discard','← 丢弃','',!ok('discard'))+
  steps.map(k=>B(k,'step '+k,'',!ok(k))).join('')+B('s','S 成功','',!ok('s')||fail)+B('f','F 失败','move',!ok('f'))+B('esc','Esc 结束','stopbtn',!ok('esc'));
 $('cStart').disabled=CBUSY||!!st?.running||!!st?.pane_busy||!st;$('cYes').disabled=CBUSY||!st?.pane_busy;
 const pg=st?.progress;$('cProg').innerHTML=!pg?'':[
  [`${pg.good} / ${pg.per}`,`当前${esc(lv?.item_label||'物品')}「${esc(pg.name||pg.cur)}」合格条数`],[pg.episodes,'这个数据集已保存'],
  pg.last?[`第 ${pg.last.episode} 条`,`最近保存：${pg.last.result==='success'?'成功':pg.last.result==='failure'?'失败':'—'}${(pg.last.issues||[]).length?' · ⚠ '+esc(pg.last.issues.join('；')):''}`]:null,
  pg.next?[esc(pg.next.k+' · '+pg.next.name),'下一个没采满的']:null].filter(Boolean).map(([b,sp])=>`<div class="k"><b style="font-size:17px">${b}</b><span>${sp}</span></div>`).join('');
 const T=$('cTerm'),atEnd=T.scrollTop+T.clientHeight>=T.scrollHeight-20;T.textContent=st?.pane||CS?.ctl_error||'';if(atEnd)T.scrollTop=T.scrollHeight;
 scPaintAll()}
async function cAct(a){if(a==='esc'&&!confirm(cMode()==='rec'?'录制中按 Esc：这一条不保存，并结束采集。确定？':'结束采集？（复位阶段按 Esc 会先保存这一条）'))return;
 CBUSY=true;cPaint();try{const r=await (await fetch('/api/collect',{method:'POST',body:JSON.stringify({action:a,preset:v('cPre')})})).json();$('cMsg').textContent=r.msg;if(!r.ok)$('cMsg').style.color='var(--bad)';else $('cMsg').style.color=''}
 catch(e){$('cMsg').textContent=String(e)}CBUSY=false;cPoll()}
function cKey(e){const k=e.key, m=cMode();let a=null;
 if(k==='ArrowRight')a='next';else if(k==='ArrowLeft')a='discard';else if(k==='Escape')a='esc';
 else if(/^[1-6]$/.test(k))a=k;else if(k.toLowerCase()==='s')a='s';else if(k.toLowerCase()==='f')a='f';
 if(!a||!(CALLOW[m]||[]).includes(a))return;e.preventDefault();cAct(a)}
let SCK=null;
function scPaintAll(){const S=CS?.scene;if(!S){$('scNow').innerHTML=esc(CS?.scene_error||'');return}
 $('scEvery').textContent=S.rotate_every;
 const key=S.dataset+':'+S.scenes.map(x=>x.id).join(',');
 if(SCK!==key){SCK=key;const cur=S.current?.scene;$('scSel').innerHTML=S.scenes.map(x=>`<option value="${esc(x.id)}">${esc(x.id)} · ${esc(x.name)}</option>`).join('')||'<option value="">（场景表里没有这个任务）</option>';if(cur)$('scSel').value=cur}
 const tot=S.scenes.reduce((a,x)=>a+(x.target||0),0), gd=S.scenes.reduce((a,x)=>a+x.good,0);
 $('scTbl').innerHTML='<tr><th>场景</th><th>桌上的工具（轮换角标）</th><th>固定的目标</th><th>合格 / 计划</th><th>已录</th><th>下一条自动摆法</th><th>同桌可顺带采</th></tr>'+
  S.scenes.map(x=>`<tr class="${S.current?.scene===x.id?'cur':''}"><td><b>${esc(x.id)}</b> ${esc(x.name)}</td><td>${esc((x.tools||[]).join('、')||'—')}</td><td>${esc((x.fixed||[]).join('、'))}</td>`+
  `<td class="${x.good>=(x.target||0)&&x.target?'ok':''}">${x.good} / ${x.target||0}</td><td>${x.all}</td><td>${x.auto_rot+1}：${esc(x.layout.map(l=>l.tool+'@'+l.slot).join('、')||'—')}</td><td>${esc((x.pair||[]).map(p=>TASKN[p]||p).join('、'))}</td></tr>`).join('')+
  `<tr><td><b>合计</b></td><td></td><td></td><td>${gd} / ${tot}${S.total&&S.total!==tot?'（任务计划 '+S.total+'）':''}</td><td colspan="3" class="na">${Object.keys(S.unknown||{}).length?'场景表里没有的：'+esc(Object.keys(S.unknown).join('、')):''}</td></tr>`;
 scPaint();
 const ci=$('ciSel');if(document.activeElement!==ci){const items=CS.items||{};$('ciLb').textContent=lvLabel();
  ci.innerHTML=Object.entries(items).map(([k,n])=>`<option value="${esc(k)}">${esc(k)} · ${esc(n||'未命名')}${String(CS.item_cur)===k?'（当前）':''}</option>`).join('')||'<option value="0">（没有物品栏）</option>';
  if(CS.item_cur)ci.value=String(CS.item_cur)}}
function lvLabel(){const t=COLL.find(c=>c.key===CS?.preset);return (t&&t.item_label)||'物品'}
function scPaint(){const S=CS?.scene;if(!S)return;const c=S.current, x=S.scenes.find(s=>s.id===v('scSel'));
 const cur=c?`<div class="s" style="font-size:22px">当前：${esc(c.scene)} ${esc(c.name||'')} · 摆法 ${(+c.rot||0)+1}</div><div class="lay">${(c.layout||[]).map(l=>`<span>${esc(l.tool)} → <b>${esc(l.slot)}</b></span>`).join('')||'<span>没有要轮换的工具</span>'}${(c.fixed||[]).map(f=>`<span>${esc(f)}：原位</span>`).join('')}</div><div class="p">设置于 ${esc(c.set||'')}</div>`
  :'<div class="s" style="font-size:20px">还没有设当前场景</div>';
 const nxt=x&&(!c||c.scene!==x.id||v('scRot')!=='auto')?`<div class="p" style="margin-top:8px">选中的 ${esc(x.id)}，自动摆法 ${x.auto_rot+1}：${esc(x.layout.map(l=>l.tool+'@'+l.slot).join('、')||'—')}（点「设为当前场景」生效）</div>`:'';
 const due=c&&x&&c.scene===x.id&&+c.rot!==x.auto_rot?`<div class="p" style="color:var(--warn);margin-top:6px">⚠ 这个场景已录 ${x.all} 条，按每 ${S.rotate_every} 条轮换该换摆法 ${x.auto_rot+1} 了</div>`:'';
 $('scNow').innerHTML=cur+nxt+due}
async function scSet(clear=false){const S=CS?.scene;if(!S)return;
 const r=await (await fetch('/api/scene',{method:'POST',body:JSON.stringify({preset:S.preset,scene:clear?'':v('scSel'),rot:v('scRot')})})).json();$('scMsg').textContent=r.msg;cPoll()}
async function ciSet(){const S=CS?.scene;if(!S)return;const r=await (await fetch('/api/collect_item',{method:'POST',body:JSON.stringify({dataset:S.dataset,current:v('ciSel')})})).json();$('ciMsg').textContent=r.msg;cPoll()}
async function loadScenesFor(){try{const j=await (await fetch('/api/scenes?task='+encodeURIComponent(v('task')))).json();const sel=$('lScene'),old=sel.value;
 sel.innerHTML='<option value="">（不填）</option>'+j.scenes.map(x=>`<option value="${esc(x.id)}">${esc(x.id)} · ${esc(x.name)}</option>`).join('');sel.value=old}catch(e){}}
setInterval(cPoll,500);
(async()=>{await loadTasks();let m='test';try{m=localStorage.getItem('nero_mode')||'test'}catch(e){}setMode(m);
 loadRuns();poll();setInterval(poll,700);evLoad();loadPlan();fanBox();loadScenesFor();setInterval(loadPlan,5000)})();
</script></body></html>"""


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    print(f"NERO 采集与测试台：http://127.0.0.1:{PORT}  （报告服务 {REPORT}；Ctrl+C 退出）", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        stop_job("网页服务退出")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
