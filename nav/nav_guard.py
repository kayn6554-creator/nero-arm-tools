"""导航安全检查（只检查、不改导航）。潘鹤 2026-10-29。
实测（navlog_1029_100258）：驱动执行 move_j 时关节基本是一个接一个转的（先关节1，再关节2……），
不是同时插值。所以要按"依次转"模拟整条路径：关节2 转完、关节4 还没伸开时，末端会先降到台面。
同时也模拟"同步插值"，两种走法里末端、肘、腕都不能低于台面 + 余量，否则拒绝这次 Move。"""
import json, math
from pathlib import Path
import numpy as np

CFG = dict(table_plane=[0.0157, -0.0297, -0.0423],   # 台面 z = a + b·x + c·y（base_link，10-28 点 5 个点拟合）
           min_tcp_above_table_m=0.05, min_link_above_table_m=0.04, steps=30)
p = Path(__file__).with_name("nav_guard.json")
if p.exists(): CFG.update(json.loads(p.read_text(encoding="utf-8")))


def _table(x, y):
    a, b, c = CFG["table_plane"]; return a + b * x + c * y


def _origins(mdh, q):
    T = np.eye(4); out = []
    for (d, a, al, off), qi in zip(mdh, q):
        th = qi + off; ca, sa, ct, st = math.cos(al), math.sin(al), math.cos(th), math.sin(th)
        T = T @ np.array([[ct, -st, 0, a], [ca * st, ca * ct, -sa, -sa * d], [sa * st, sa * ct, ca, ca * d], [0, 0, 0, 1]])
        out.append(T[:3, 3].copy())
    return out


def _clear(ik, q):
    tcp, _ = ik.fk(q); org = _origins(ik.mdh, q)
    t = min(p[2] - _table(p[0], p[1]) for p in (tcp, org[-1]))
    l = min(p[2] - _table(p[0], p[1]) for p in org[2:-1])
    return t, l


def check(ik, start, goal):
    """不通过就抛 RuntimeError（网页会显示原因、不发指令）。通过返回最低余量。"""
    s, g = np.asarray(start, float), np.asarray(goal, float)
    worst = {}
    for name in ("依次转（实测）", "同时转"):
        tmin = lmin = 9.0; where = ""
        q = s.copy()
        if name == "同时转":
            for t in np.linspace(0, 1, CFG["steps"] * 2):
                a, b = _clear(ik, s + (g - s) * t)
                if a < tmin: tmin, where = a, f"走到 {t:.0%}"
                lmin = min(lmin, b)
        else:
            for j in range(7):
                for t in np.linspace(0, 1, CFG["steps"]):
                    qq = q.copy(); qq[j] = s[j] + (g[j] - s[j]) * t
                    a, b = _clear(ik, qq)
                    if a < tmin: tmin, where = a, f"关节{j + 1} 转的时候"
                    lmin = min(lmin, b)
                q[j] = g[j]
        worst[name] = (tmin, lmin)
        if tmin < CFG["min_tcp_above_table_m"]:
            raise RuntimeError(f"安全检查：按{name}模拟，{where}末端离台面只有 {tmin * 100:.1f} cm（要求 ≥ {CFG['min_tcp_above_table_m'] * 100:.0f} cm），拒绝执行。先 Go home 再点，或者换个点")
        if lmin < CFG["min_link_above_table_m"]:
            raise RuntimeError(f"安全检查：按{name}模拟，肘/腕离台面只有 {lmin * 100:.1f} cm，拒绝执行。先 Go home 再点")
    return {k: dict(tcp_min_cm=round(v[0] * 100, 1), link_min_cm=round(v[1] * 100, 1)) for k, v in worst.items()}
