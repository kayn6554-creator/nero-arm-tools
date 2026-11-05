#!/usr/bin/env python3
# 常驻软件急停：提前连好服务，按回车 = 关控制锁 + 保持当前位置
import sys
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool, Empty

rclpy.init()
node = Node("panhe_estop")
gate = node.create_client(SetBool, "/control_enable")
hold = node.create_client(Empty, "/emergency_stop")
for client, name in [(gate, "/control_enable"), (hold, "/emergency_stop")]:
    if not client.wait_for_service(timeout_sec=3.0):
        print(f"找不到服务 {name}，驱动是否在运行？")
        sys.exit(1)

print("\n=== 软件急停已就绪 ===")
print("按 回车 = 关锁 + 保持当前位置；输入 q 再回车 = 退出\n")
while True:
    line = sys.stdin.readline().strip()
    if line == "q":
        break
    req = SetBool.Request()
    req.data = False
    f_hold = hold.call_async(Empty.Request())
    f_gate = gate.call_async(req)
    rclpy.spin_until_future_complete(node, f_hold, timeout_sec=2.0)
    rclpy.spin_until_future_complete(node, f_gate, timeout_sec=2.0)
    print(">>> 急停已发送 | 保持:", "已执行" if f_hold.done() else "无响应",
          "| 控制锁:", f_gate.result().message if f_gate.result() else "无响应")
node.destroy_node()
rclpy.shutdown()
