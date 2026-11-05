#!/usr/bin/env python3
"""NERO 模型测试台（网页）：把 nero_vla_client.py / nero_replay_chunk.py 和急停、关锁、夹爪做成按钮。

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
TASKS = {"weigh": "称重", "task2": "水温测量", "task3": "风速测量"}

lock = threading.Lock()
log = deque(maxlen=1500)
job = {"name": None, "proc": None, "rc": None, "started": 0.0, "ended": 0.0}
pending_meta = {}     # 下一次 live 的物品 / 视频编号 / 是否结束后回到主臂姿态，看到 RUN_DIR= 时写进那次的记录
last_estop = [0.0]    # 最近一次按急停的时刻：急停后不自动动手臂


def add_log(line: str):
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
    task = task_of(p)
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
        ok, msg = start_job(f"空跑 dry × {n}", EVO +
                            f"python -u ~/nero_vla_client.py --mode dry --server-port {port} --task {task} --max-chunks {n} --dry-interval-s 0.5")
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
        pending_meta.clear()
        pending_meta.update(task=task, item=str(p.get("item", ""))[:40], video=str(p.get("video", ""))[:60], auto_return=auto)
        t_start = time.time()

        def after(rc):   # 运行结束：没按过急停、勾了“自动回到主臂姿态”，就 2 秒后自动执行
            if not auto:
                return
            if last_estop[0] >= t_start:
                add_log("── 运行中按过急停，不自动回到主臂姿态（确认安全后手动点）")
                return
            time.sleep(2.0)
            add_log("── 自动：从臂回到主臂姿态")
            start_job("从臂回到主臂姿态（自动）", EVO + "python -u ~/nero_goto_leader.py --gripper-mm 93 --speed-deg 15", stdin_text="yes\n")
        ok, msg = start_job(f"自动运行 live 最多 {n} 块 / {t} s", EVO +
                            f"python -u ~/nero_vla_client.py --mode live --server-port {port} --task {task} "
                            f"--exec-steps {e} --slow {s:g} --max-chunks {n} --max-duration-s {t}", stdin_text="yes\n", on_done=after)
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
                    "exec_steps": s["args"].get("exec_steps"), "slow": s["args"].get("slow"), "evaluated": run in done})
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
            msg = handle_action(body.get("action", ""), body.get("params", {}))
            self._send(200, {"ok": True, "msg": msg})
        except Exception as e:
            self._send(400, {"ok": False, "msg": str(e)})


PAGE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NERO 模型测试台</title>
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
dialog{border:none;border-radius:12px;padding:20px;max-width:440px}dialog::backdrop{background:#0006}
dialog h3{margin:0 0 8px}dialog ul{padding-left:18px;margin:8px 0;color:#333}
</style></head><body>
<header>
 <h1>NERO 模型测试台</h1><span id="chip" class="chip">空闲</span>
 <label>任务</label><select id="task"><option value="weigh">称重</option><option value="task2">水温测量</option><option value="task3">风速测量</option></select>
 <label>模型端口</label><input id="port" value="6699">
 <span class="sp"></span>
 <button id="bStop" onclick="act('stop')">停止当前任务</button>
 <button onclick="act('lock')">关锁</button>
 <button class="estop" onclick="act('estop')">软件急停</button>
</header>
<main>
 <section class="card wide"><h2>⓪ 拖主臂把从臂带到物品附近<span class="tag">两台手臂会动</span></h2>
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
 <section class="card"><h2>① 准备<span class="tag safe">不动手臂</span></h2>
  <p>读状态：使能和限位。check：不连模型，检查关节、两相机能否读到。</p>
  <div class="row"><button class="job" onclick="act('status')">读从臂状态</button><button class="job" onclick="act('check')">检查 check</button></div>
  <div class="row"><label>夹爪开到</label><input id="gw" value="93"><label>mm</label>
   <button class="job move" onclick="act('gripper',{width_mm:v('gw')})">设置夹爪</button><span class="tag">只动夹爪</span></div>
 </section>
 <section class="card"><h2>② 空跑 dry<span class="tag safe">不动手臂</span></h2>
  <p>连模型推理，只记录动作块不执行。手臂静止时模型常先“等” 2 秒，多采几块挑有动作的。</p>
  <div class="row"><label>采</label><input id="dn" value="8"><label>块</label><button class="job pri" onclick="act('dry',{chunks:v('dn')})">开始空跑</button></div>
 </section>
 <section class="card wide"><h2>③ 查看动作块 · 手动执行<span class="tag">执行时手臂会动</span></h2>
  <p>选一次 dry 记录和其中一块，看每一步相对当前姿态的变化（°），再只执行前 N 步。执行前会检查手臂与记录时相差 ≤ 2°。</p>
  <div class="row"><label>记录</label><select id="run" onchange="loadChunks()"></select><button onclick="loadRuns()">刷新</button>
   <label>块</label><select id="chunk" onchange="drawTable()"></select>
   <label>执行前</label><input id="rn" value="10" oninput="drawTable()"><label>步</label><label>放慢</label><input id="rs" value="3"><label>倍</label>
   <button class="job move" onclick="replay()">手动执行</button></div>
  <div class="summary" id="sum"></div>
  <div class="tbl"><table id="tbl"></table></div>
 </section>
 <section class="card wide"><h2>④ 自动运行 live<span class="tag">手臂会动</span></h2>
  <p>模型闭环控制：推理 → 停稳 → 执行，循环。整块执行(30 步)才能跳过开头的等待。</p>
  <div class="row"><label>每块执行</label><input id="le" value="30"><label>步</label><label>放慢</label><input id="ls" value="3"><label>倍</label></div>
  <div class="row"><label id="lItemLb">物品</label><select id="lItem"></select><label>外接相机视频编号</label><input id="lVideo" style="width:150px" placeholder="如 VID_0012">
   <label style="color:var(--ink)"><input type="checkbox" id="lAuto" style="width:auto"> 结束后自动回到主臂姿态</label><span class="summary">（按过急停就不会自动动）</span></div>
  <div class="row"><label>最多</label><input id="ln" value="40"><label>块</label><label>最长</label><input id="lt" value="600"><label>秒</label>
   <button class="job move" onclick="confirmRun('live',{exec_steps:v('le'),slow:v('ls'),chunks:v('ln'),max_s:v('lt'),item:v('lItem'),video:v('lVideo'),auto_return:$('lAuto').checked?1:0},'自动运行：每块执行 '+v('le')+' 步、放慢 '+v('ls')+' 倍，最多 '+v('ln')+' 块 / '+v('lt')+' 秒。运行中随时可按顶部“软件急停”或“停止当前任务”。')">开始自动运行</button></div>
 </section>
 <section class="card wide" id="evalCard"><h2>⑤ 评测：每次自动运行后人工打勾<span class="tag safe">不动手臂</span></h2>
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
 <section class="card wide"><h2>日志</h2><pre id="log"></pre></section>
</main>
<dialog id="dlg"><h3>确认：手臂将运动</h3><div id="dlgText"></div>
 <ul><li>手放实体急停</li><li>从臂周围已清空，离台面 / 物体 ≥ 10 cm</li><li>采集录制程序没在运行，定位网页没在 Move</li></ul>
 <label><input type="checkbox" id="dlgOk" style="width:auto"> 以上都已确认</label>
 <div class="row" style="justify-content:flex-end;margin-top:12px"><button onclick="dlg.close()">取消</button><button class="move" id="dlgGo">执行</button></div></dialog>
<script>
const $=id=>document.getElementById(id), v=id=>$(id).value.trim();
let since=0, chunks=null, running=false;
async function post(action,params={}){params.task=v('task');params.port=v('port');
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
  if(wasRunning&&/自动运行/.test(s.name)){if(v('lVideo'))$('lVideo').value=bumpVideo(v('lVideo'));evLoad(true).then(()=>{const c=$('evalCard');c.classList.remove('flash');void c.offsetWidth;c.classList.add('flash');c.scrollIntoView({behavior:'smooth'})})}}
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
 $('evTbl').innerHTML='<tr><th>次</th><th>开始</th><th>'+esc(I.item_label)+'</th><th>视频</th><th>块 / 用时</th><th>结束原因</th>'+steps.map(([k,nm])=>`<th title="${esc(nm)}">${k}</th>`).join('')+'<th>整体</th><th>备注</th><th></th></tr>'+
  runs.map(r=>{const e=byRun[r.run];return `<tr class="${r.run===v('evRun')?'cur':''}"><td><b>${r.no}</b></td><td>${esc(String(r.started).slice(5,16))}</td><td>${esc(r.item||'')}</td><td>${esc(r.video||'')}</td><td>${r.chunks} 块 · ${r.duration_s ?? '?'} s</td><td class="na">${esc(r.end_reason||'')}</td>`+
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
$('task').addEventListener('change',()=>evLoad());
loadRuns();poll();setInterval(poll,700);evLoad();
</script></body></html>"""


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    print(f"NERO 模型测试台：http://127.0.0.1:{PORT}  （Ctrl+C 退出）", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        stop_job("网页服务退出")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
