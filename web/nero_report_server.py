#!/usr/bin/env python3
"""报告网站服务：提供 lerobot_data 目录下的静态文件 + 一个保存 step 分界修正的接口。
修正只写到 <数据集>/report/step_edits.json，不碰数据集本身；写进数据集用 ~/nero_steps_apply.py。
用法：python ~/nero_report_server.py /data/panhe/lerobot_data 8766"""
import datetime, functools, json, os, re, subprocess, sys, threading, time, urllib.parse, urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nero_report_lib as NR

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/data/panhe/lerobot_data").resolve()
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8766
LOCK = threading.Lock()
CTL_LOCK = threading.Lock()      # 控制按钮单独一把锁，定位开关要几秒，不挡住改标签
# ---- 采集控制：网页按钮 → tmux 把按键送进录制程序所在的终端（窗口 0 上格）；定位开关直接调 pos_on/pos_off ----
TMUX = os.environ.get("NERO_TMUX", "nero:0.0")
HOME = Path.home()
KEYS = {"next": "Right", "discard": "Left", "esc": "Escape", "s": "s", "f": "f", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6"}
ALLOWED = {"prep": {"next", "esc"}, "armed": {"next", "esc"}, "start": {"esc"}, "rec": {"1", "2", "3", "4", "5", "6", "s", "f", "discard", "esc"},
           "reset": {"next", "discard", "esc"}, "busy": set()}


def _tmux(*a):
    return subprocess.run(["tmux", *a], capture_output=True, text=True, timeout=5)


def _recording():
    return subprocess.run(["pgrep", "-f", r"python[0-9.]* .*nero_record\.py"], capture_output=True).returncode == 0


def _pane_busy():
    """终端里的 shell 有没有子进程在跑（nero_record.sh 是 bash 脚本，看 pane_current_command 分不出来）。"""
    pid = _tmux("display-message", "-p", "-t", TMUX, "#{pane_pid}").stdout.strip()
    return bool(pid) and subprocess.run(["pgrep", "-P", pid], capture_output=True).returncode == 0


def _live():
    try:
        d = json.loads((ROOT / "_live.json").read_text(encoding="utf-8"))
        d["fresh"] = time.time() - float(d.get("t", 0)) < 3
        return d
    except Exception:
        return None


def _ros_env():
    env = {k: v for k, v in os.environ.items() if k.startswith(("ROS_", "RMW_", "FASTRTPS", "CYCLONEDDS"))}
    env.update(HOME=str(HOME), PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", LANG="C.UTF-8")
    return env


def ctl_state():
    pane, cmd = "", ""
    r = _tmux("capture-pane", "-p", "-J", "-t", TMUX, "-S", "-60")
    if r.returncode == 0:
        lines = [ln.rstrip() for ln in r.stdout.splitlines()]
        while lines and not lines[-1]: lines.pop()
        pane = "\n".join(lines[-28:])
        cmd = _tmux("display-message", "-p", "-t", TMUX, "#{pane_current_command}").stdout.strip()
        busy = _pane_busy()
    else:
        busy = None
        pane = f"读不到 tmux 终端 {TMUX}：{r.stderr.strip()}"
    live = _live()
    return dict(ok=True, running=_recording(), live=live, pane=pane, pane_cmd=cmd, pane_busy=busy, target=TMUX,
                progress=_progress(live["dataset"]) if live and live.get("dataset") else None)


def _progress(name):
    """正在录的数据集：当前物品完成几条（成功且没问题）、下一个没采满的物品、刚保存的那一条有没有问题。"""
    if not re.fullmatch(r"[\w.\-]+", name): return None
    lab = NR.load_labels(ROOT, name); eps = []
    try:
        eps = json.loads((ROOT / name / "report" / "summary.json").read_text(encoding="utf-8")).get("items", [])
    except Exception:
        pass
    per, cur, keys = int(lab.get("per_item") or 10), str(lab.get("current") or 0), list(lab.get("items", {}))
    fail = NR.fail_cfg(NR.task_for_dataset(name)[1])
    res_ok = "failure" if fail else "success"
    good = lambda k: sum(1 for e in eps if str(e.get("item")) == k and e.get("result") == res_ok and not e.get("issues"))
    nxt = None
    if keys and not fail:
        i0 = keys.index(cur) if cur in keys else -1
        for j in range(1, len(keys) + 1):
            k = keys[(i0 + j) % len(keys)]
            if k != cur and good(k) < per: nxt = dict(k=k, name=lab["items"][k] or f"{k} 号"); break
    last = eps[-1] if eps else None
    fp = None
    if fail:   # 失败数据：按失败类型计数，建议下一条做最少的那种
        pt = int(fail.get("per_type", 20))
        ty = [dict(k=k, name=v, good=sum(1 for e in eps if str(e.get("fail_type")) == k and not e.get("issues")),
                   total=sum(1 for e in eps if str(e.get("fail_type")) == k)) for k, v in fail["types"].items()]
        todo = [t for t in ty if t["good"] < pt]
        fp = dict(per=pt, types=ty, suggest=min(todo, key=lambda t: (t["good"], int(t["k"])))["k"] if todo else None)
    return dict(fail=fp, cur=cur, name=lab.get("items", {}).get(cur, ""), good=good(cur), total=sum(1 for e in eps if str(e.get("item")) == cur),
                per=per, next=nxt, episodes=len(eps),
                last=dict(episode=last["episode"], result=last.get("result"), issues=last.get("issues", []), item_name=last.get("item_name"), fail_type=last.get("fail_type", 0),
                          recorded=last.get("recorded")) if last else None)


# ---- 一键处理并导出：写入修改 → 逐条核对 → 打包，日志给网页看 ----
JOBS = {}


def _export(name):
    job = JOBS[name]; root = ROOT / name; home = str(HOME)
    def log(x): job["log"].append(x)
    def run(cmd, cwd=None):
        log("$ " + cmd)
        p = subprocess.Popen(["bash", "-c", f"source {home}/miniconda3/etc/profile.d/conda.sh 2>/dev/null; source {home}/env_evo.sh 2>/dev/null; {cmd}"],
                             cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out = []
        for line in p.stdout:
            line = line.rstrip(); out.append(line); log(line)
        return p.wait(), out
    try:
        job["stage"] = "写入修改"
        rc, out = run(f"python {home}/nero_steps_apply.py {root}")
        if rc != 0 and not any("没有待写入" in x for x in out): raise RuntimeError("写入修改失败，没有继续")
        job["stage"] = "逐条核对"
        rc, out = run(f"python {home}/nero_verify.py {root}")
        if rc != 0 or not out or "全部通过" not in out[-1]:
            raise RuntimeError("核对没有全部通过，没有打包。先在网页上处理问题条（或删掉重录）再导出")
        job["stage"] = "打包"
        tgz = f"{home}/{name}.tgz"
        extra = f" _labels/{name}.json" if (ROOT / "_labels" / f"{name}.json").exists() else ""
        rc, _ = run(f"rm -f {tgz} && tar czf {tgz} --exclude='report/backup_*' --exclude='depth/.tmp' {name}{extra} && ls -lh {tgz}", cwd=str(ROOT))
        if rc != 0: raise RuntimeError("打包失败")
        job.update(status="done", stage="完成", message=f"已打包：{tgz}。Windows 下载：scp panhe@192.168.1.102:~/{name}.tgz %USERPROFILE%\\Downloads\\")
    except Exception as ex:
        job.update(status="failed", message=str(ex))
    finally:
        job["ended"] = datetime.datetime.now().strftime("%H:%M:%S")


def export_start(name):
    if not re.fullmatch(r"[\w.\-]+", name) or not (ROOT / name / "meta" / "info.json").exists(): raise ValueError("没有这个数据集")
    if _recording(): raise ValueError("录制程序正在运行，先按 Esc 结束录制再导出")
    if JOBS.get(name, {}).get("status") == "running": raise ValueError("已经在导出了")
    JOBS[name] = dict(status="running", stage="开始", log=[], message="", started=datetime.datetime.now().strftime("%H:%M:%S"))
    threading.Thread(target=_export, args=(name,), daemon=True).start()
    return "开始处理：写入修改 → 核对 → 打包"


def ctl(req):
    a = str(req.get("action", ""))
    running, live = _recording(), _live()
    mode = (live or {}).get("mode") if live and live.get("fresh") else None
    if a == "start":
        key = str(req.get("preset", ""))
        t = NR.load_tasks().get(key)
        if not t or not t.get("task"): raise ValueError(f"没有这个任务预设：{key}")
        if running: raise ValueError("录制程序已经在运行")
        if _tmux("has-session", "-t", TMUX.split(":")[0]).returncode != 0: raise ValueError(f"没有 tmux 会话 {TMUX.split(':')[0]}")
        if _pane_busy(): raise ValueError(f"终端 {TMUX} 里还有程序在运行，先在那里结束它")
        _tmux("send-keys", "-t", TMUX, f"source ~/env_evo.sh && ~/nero_record.sh {key}", "Enter")
        return f"已在终端 {TMUX} 启动 ~/nero_record.sh {key}，开录前检查要十几秒"
    if a == "yes":
        if not _pane_busy(): raise ValueError("终端里没有程序在运行，不需要输入 yes")
        _tmux("send-keys", "-t", TMUX, "yes", "Enter"); return "已输入 yes"
    if a in ("pos_on", "pos_off"):
        if not running: raise ValueError("录制程序没在运行：定位模式由录制程序里的从臂插件提供")
        if a == "pos_on" and mode == "rec": raise ValueError("录制中不能开定位（会把这一条的阶段打乱）；先 s 结束或 ← 丢弃")
        if a == "pos_on" and mode == "armed": raise ValueError("就绪阶段不能开定位：定位会带动主臂，马上触发开录。先在复位阶段定位好，再按 →")
        if a == "pos_off":
            try:
                st = json.loads(urllib.request.urlopen("http://127.0.0.1:8765/api/state", timeout=1).read())
                if (st.get("motion") or {}).get("state") == "moving": raise ValueError("网页显示从臂还在移动，等 Target reached 2 秒后再关定位")
            except ValueError:
                raise
            except Exception:
                pass
        r = subprocess.run(["bash", str(HOME / f"{a}.sh")], capture_output=True, text=True, timeout=20, env=_ros_env())
        out = (r.stdout + r.stderr).strip()
        if r.returncode != 0 or not out: raise ValueError(f"{a}.sh 没有成功：{out or '没有输出（服务没响应？）'}")
        return f"{'定位开' if a == 'pos_on' else '定位关'}：{out}"
    if a in KEYS:
        if not running: raise ValueError("录制程序没在运行")
        if mode is None: raise ValueError("读不到录制程序的状态（_live.json 没更新），请在终端里按键")
        if a not in ALLOWED.get(mode, set()): raise ValueError(f"现在是“{ {'prep': '准备', 'armed': '就绪（动主臂就开始录）', 'rec': '录制中', 'reset': '复位', 'busy': '保存中', 'start': '启动中'}.get(mode, mode) }”阶段，这个键无效")
        if a in "123456" and str(a) not in {str(k) for k in (live.get("steps") or {})}: raise ValueError(f"这个任务没有 step {a}")
        if a == "s" and live.get("fail"): raise ValueError("失败数据集只用 f 结束：停在失败的那一步按 f")
        _tmux("send-keys", "-t", TMUX, KEYS[a]); return f"已发送 {KEYS[a]}"
    raise ValueError(f"不认识的操作：{a}")




class H(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *a):
        if self.command == "POST": super().log_message(fmt, *a)

    def end_headers(self):
        if self.path.split("?")[0].endswith((".json", ".html", "/")):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _reply(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path.split("?")[0] == "/api/export_state":
            q = dict(x.split("=", 1) for x in (self.path.split("?", 1) + [""])[1].split("&") if "=" in x)
            j = JOBS.get(urllib.parse.unquote(q.get("dataset", "")))
            return self._reply(200, dict(ok=True, job={**j, "log": j["log"][-200:]} if j else None))
        if self.path.split("?")[0] == "/api/ctl_state":
            try:
                return self._reply(200, ctl_state())
            except Exception as ex:
                return self._reply(500, dict(ok=False, error=str(ex)))
        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/labels":
            return self._labels()
        if self.path == "/api/export":
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0))
                return self._reply(200, dict(ok=True, message=export_start(str(req.get("dataset", "")))))
            except Exception as ex:
                return self._reply(400, dict(ok=False, error=str(ex)))
        if self.path == "/api/ctl":
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0))
                with CTL_LOCK:
                    msg = ctl(req)
                return self._reply(200, dict(ok=True, message=msg))
            except Exception as ex:
                return self._reply(400, dict(ok=False, error=str(ex)))
        if self.path != "/api/steps":
            return self._reply(404, dict(ok=False, error="unknown api"))
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0))
            name, ep, bounds = str(req["dataset"]), int(req["episode"]), req.get("bounds")
            if not re.fullmatch(r"[\w.\-]+", name): raise ValueError("数据集名不合法")
            ds = ROOT / name; rep = ds / "report"
            e = NR.load_episode_json(rep, ep)
            if e is None: raise ValueError(f"找不到第 {ep} 条")
            L = float(e["length_s"])
            if bounds is not None:
                bounds = [round(float(b), 3) for b in bounds]
                NR.use_task(name)
                if NR.fail_cfg(NR.task_for_dataset(name)[1]):    # 失败数据：step 到失败那一步为止，分界可以少
                    if len(bounds) > len(NR.STEP_NAMES) - 1: raise ValueError(f"最多 {len(NR.STEP_NAMES) - 1} 个分界")
                elif len(bounds) != len(NR.STEP_NAMES) - 1: raise ValueError(f"需要 {len(NR.STEP_NAMES) - 1} 个分界")
                if not all(a < b for a, b in zip([0.0] + bounds, bounds + [L])):
                    raise ValueError("分界必须从小到大，且在 0 和本条时长之间")
            with LOCK:
                edits = NR.load_edits(rep)
                if bounds is None: edits.pop(str(ep), None)
                else: edits[str(ep)] = dict(bounds=bounds, updated=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), applied=False)
                NR.save_edits(rep, edits)
                NR.build_site(ds)
            self._reply(200, dict(ok=True))
        except Exception as ex:
            self._reply(400, dict(ok=False, error=str(ex)))


    def _labels(self):
        """{dataset, items?: {号: 名字}, current?: 号, episode?: 条, item?: 号}"""
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0))
            name = str(req["dataset"])
            if not re.fullmatch(r"[\w.\-]+", name): raise ValueError("数据集名不合法")
            with LOCK:
                lab = NR.load_labels(ROOT, name)
                if "items" in req:
                    items = {str(int(k)): str(v).strip()[:40] for k, v in req["items"].items() if int(k) > 0}
                    lab["items"] = dict(sorted(items.items(), key=lambda kv: int(kv[0])))
                if "current" in req:
                    c = int(req["current"])
                    if c and str(c) not in lab["items"]: raise ValueError("没有这个物品号")
                    lab["current"] = c
                if "episodes_map" in req:     # 批量：{条: 物品号}
                    for ep_, it_ in req["episodes_map"].items():
                        it_ = int(it_ or 0)
                        if it_ and str(it_) not in lab["items"]: raise ValueError("没有这个物品号")
                        lab["episodes"][str(int(ep_))] = it_
                if "episode" in req:
                    ep, it = int(req["episode"]), int(req.get("item") or 0)
                    if it and str(it) not in lab["items"]: raise ValueError("没有这个物品号")
                    lab["episodes"][str(ep)] = it
                NR.save_labels(ROOT, name, lab)
                if (ROOT / name / "report").exists(): NR.build_site(ROOT / name)
                else: NR.write_index(ROOT)
            self._reply(200, dict(ok=True, labels=lab))
        except Exception as ex:
            self._reply(400, dict(ok=False, error=str(ex)))


if __name__ == "__main__":
    NR.write_index(ROOT)
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), functools.partial(H, directory=str(ROOT)))
    print(f"报告网站：http://127.0.0.1:{PORT}/    数据目录：{ROOT}    （Ctrl+C 停止）", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
