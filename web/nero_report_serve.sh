#!/bin/bash
# 用法：~/nero_report_serve.sh        （Windows 需要 ssh -N -L 8766:127.0.0.1:8766 panhe@192.168.1.102）
# 打开 http://127.0.0.1:8766/ 就是数据总览；可以在网页里修正 step 分界
~/miniconda3/envs/evo-rl/bin/python ~/nero_report_server.py /data/panhe/lerobot_data 8766
