#!/usr/bin/env python3
"""只连模型服务读 metadata（不读相机、不动手臂），给测试台的「检查连接」用。

    python ~/nero_vla_ping.py 172.16.11.113 6706
输出一行 PING={...} 或 PING_FAIL={...}，测试台据此显示连接状态和夹爪模式。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    host, port = sys.argv[1], int(sys.argv[2])
    try:
        from nero_vla_client import PolicyClient
        c = PolicyClient(host, port, timeout_s=5.0)
        md = c.metadata
        c.close()
        print("PING=" + json.dumps({"host": host, "port": port, "md": md}, ensure_ascii=False, default=str), flush=True)
        return 0
    except Exception as e:
        print("PING_FAIL=" + json.dumps({"host": host, "port": port, "err": f"{type(e).__name__}: {e}"}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
