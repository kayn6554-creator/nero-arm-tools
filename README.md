# NERO 机械臂：数据采集与 VLA 实机测评工具

在 NERO 7 轴机械臂（主从遥操作）上采集 LeRobot v3 数据集、测试 VLA 模型的代码。目标是复现 π*0.6 的 RECAP 方法（开源实现 [Evo-RL](https://github.com/MINT-SJTU/Evo-RL)）：采集 → 训练 → 实机测试 → 迭代。

本仓库只包含我们自己写的代码。机械臂驱动（agx_arm_ws、pyAgxArm）、Click-to-Reach 定位网页和 IK 来自课题组已有项目，不在这里。

## 目录

| 目录 | 内容 |
|---|---|
| `plugins/` | LeRobot 插件：从臂 `lerobot_robot_nero`、主臂 `lerobot_teleoperator_nero`、ROS 相机 `lerobot_camera_ros` |
| `record/` | 录制入口与开机检查：`nero_record.sh` / `nero_record.py`（step 按键、物品标签、10 Hz 存储、深度、两相机时间戳）、主从对齐、限位、急停、相机检查 |
| `data_tools/` | 报告与数据处理：单条统计、数据标准检查、写入修改、逐条核对、删条、任务预设、汇总页 |
| `web/` | 采集控制台（端口 8766）：网页控制录制、看进度、改 step 和物品、一键导出 |
| `nav/` | 定位网页的安全检查 `nav_guard.py`（路径离台面高度模拟）和轨迹记录 |
| `vla/` | 模型测试：客户端、测试台（端口 8767）、回到主臂姿态、动作块回放、夹爪观测 |
| `ops/` | 环境变量、一键体检 |

## 数据格式（采集）

- LeRobot v3，10 Hz（控制循环 30 Hz，每 3 帧存 1 帧）
- `observation.state` / `action`：8 维，joint1–7（rad）+ gripper（m，0–0.1）；`action` 的夹爪是主臂指令换算到从臂行程
- 头部 + 腕部 RGB 视频；对齐到彩色的深度另存为无损 mkv（`depth/<相机>/episode_XXXXXX.mkv`，uint16 毫米）
- 逐帧附加字段（不以 `observation.` 开头，不作为模型输入）：`step_index`（1–4）、`item_index`、`leader.gripper.pos`、`camera.stamp_ms`、`camera.sync_ms`、`camera.depth_rgb_ms`
- 失败数据集（`*_fail_v1`）：到失败的那一步按 f 结束，失败类型 F1–F4 = 失败发生在第几步，存在 `_labels/<数据集>.json`

## 任务

| 预设 | 数据集 | 指令 |
|---|---|---|
| weigh | nero_weigh_v1 | pick up the object and place it on the scale |
| task2 | nero_temp_v1 | pick up the thermometer and measure the water temperature in the container |
| task3 | nero_wind_v1 | pick up the anemometer and measure the wind speed |

另有对应的失败预设 `weigh_fail` / `task2_fail` / `task3_fail`。

## 依赖

Ubuntu + ROS 2 Humble、RealSense ROS 驱动、Evo-RL（LeRobot）conda 环境、NERO 驱动与 pyAgxArm。脚本里的路径和 IP 按实验室服务器写死，换机器时需要修改（见 `ops/env_evo.sh`）。

## 说明

- `nav/ik_click_deploy.sh` 部署定位网页时会用到基于原项目修改的 `app.py`、`ros_robot.py` 和 `ros_camera.py`，这几个文件不在本仓库。
- 安全约定：机械臂运动时必须有人手放实体急停；不要用关闭使能来急停。
