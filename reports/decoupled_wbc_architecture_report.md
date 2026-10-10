# Decoupled WBC 技术考古报告

> 仓库：`NVlabs/GR00T-WholeBodyControl`（本地分支 main，提交 b042411）
> 对象：`decoupled_wbc/` 子包 —— NVIDIA GR00T N1.5 / N1.6 模型所用的"解耦"全身控制器
> 一句话定义：**下半身（腿 + 腰，15 自由度）由 RL 策略闭环控制，上半身（双臂，14 自由度，可选 + 腰）由遥操作 IK 给出的关节目标做限速插值，两者在一个 50 Hz 主循环里拼装成完整关节指令下发。**

---

## 1. 目录结构总览

| 子目录 | 作用 | 关键内容 |
|---|---|---|
| `control/` | **核心控制栈**（本报告重点） | 抽象基类、G1 机器人接口、主控制循环、策略、IK、遥操作 |
| `sim2mujoco/` | 独立的 MuJoCo sim2sim 验证环境 | G1 的 MJCF/URDF、两个已发布的 ONNX RL 策略、最小可运行 demo |
| `data/` | 数据采集与导出 | 继承 HuggingFace `LeRobotDataset` 的 `Gr00tDataExporter`，GR00T 风格 modality 元数据，rerun 可视化 |
| `dexmg/` | 内嵌（vendored）的 **gr00trobocasa** —— RoboCasa 的 NVIDIA fork | 移动操作（loco-manipulation）仿真环境，用于仿真数据采集 |
| `scripts/` | 可运行入口脚本 | 一键部署编排、LeapMotion 示例、webcam 录像 |
| `docker/` | 部署镜像 | 基于内部镜像 `nvgear/ros-2`，含 Manus 手套 udev 规则、tmux 配置、entrypoint |
| `tests/` | pytest 测试（约 2800 行） | 机器人模型、IK 重定向、插值策略、数据导出、控制回路 |

包元信息：`pyproject.toml`（包名 `decoupled_wbc`，作者署名 "NVIDIA Gear Lab"）。关键依赖透露了技术选型：`pin`（Pinocchio 刚体动力学库）、`pin-pink`（Pink 任务空间 IK）、`qpsolvers[osqp,quadprog]`（QP 求解器）、`onnxruntime`（RL 策略推理）、`mujoco`、`lerobot`（数据格式）、`rerun-sdk`（可视化）、`meshcat`。

---

## 2. 整体架构：下半身 RL + 上半身 IK 如何解耦、在哪汇合

### 2.1 汇合点

汇合发生在 **`G1DecoupledWholeBodyPolicy`**（`decoupled_wbc/control/policy/g1_decoupled_whole_body_policy.py`）。它持有两个子策略：

- `upper_body_policy`：默认是 `InterpolationPolicy`（`control/policy/interpolation_policy.py`）
- `lower_body_policy`：`G1GearWbcPolicy`（`control/policy/g1_gear_wbc_policy.py`）

每个控制周期（默认 **50 Hz**），`get_action()` 做三件事：

1. 问上半身策略要 `target_upper_body_pose`（17 维：双臂 14 + 可选腰 3），写入全身关节向量 `q` 的上半身槽位；
2. **把上半身信息回馈给下半身策略**：用 Pinocchio 正运动学算出躯干相对腰部的姿态（剔除腰 yaw 后的 roll/pitch/yaw），连同手臂目标关节角 `q_arms`、基座高度命令、导航速度命令一起传给下半身策略——这是"解耦但不孤立"的关键设计：RL 策略知道上半身正在干什么，从而能保持平衡；
3. 问下半身策略要 15 维动作（腿 12 + 腰 3），写入 `q` 的下半身槽位。**腰（pelvis/waist）若同时属于两组，下半身策略优先**（稳定性优先于跟踪精度）。

最终输出 `{"q": q}`，一个 29 维（+手 14 维 = 43 维）的全身关节目标位置。

### 2.2 一个重要的反直觉事实：IK 不在主控制循环里

上半身的逆运动学发生在**另一个进程**：

```
VR 设备 (Pico / Vive / iPhone / LeapMotion / Manus 手套 / JoyCon)
   → streamer（control/teleop/streamers/）
   → TeleopPolicy（control/policy/teleop_policy.py）
   → TeleopRetargetingIK（control/teleop/teleop_retargeting_ik.py，内含 Pink IK）
   → 关节空间目标 target_upper_body_pose
   → ROS 2 话题 "ControlPolicy/upper_body_pose"（CONTROL_GOAL_TOPIC，20 Hz）
   → 主控制循环订阅 → InterpolationPolicy 做带最大关节速度限制的轨迹插值
```

也就是说，**IK 的产物是关节角而不是力矩**；主循环里的上半身"策略"本质上只是一个平滑插值器（开环，不读观测）。这样设计的好处：IK 算慢了也不会拖垮 50 Hz 的控制回路，遥操作丢帧时插值器继续沿最后轨迹走，且 1 秒超时后会注入安全目标让机器人停下。

### 2.3 关节分组（解耦的"接缝"定义处）

分组定义在 `control/robot_model/supplemental_info/g1/g1_supplemental_info.py` 的 `G1SupplementalInfo`：

- `lower_body` = `legs`（左右腿各 6 自由度）+ `waist`（3 自由度）= **15 自由度**
- `upper_body` = `arms`（左右臂各 7 自由度）+ `hands`（左右手各 7 自由度）
- `body` = lower_body + 手臂 = 29 自由度（不含手）

腰部归属可通过 `waist_location` 参数配置（`instantiate_g1_robot_model()`，`control/robot_model/instantiation/g1.py`）：
- `lower_body`：腰完全交给 RL（默认）
- `upper_body`：腰由 IK 控制（操作任务时弯腰）
- `lower_and_upper_body`：IK 算出腰的参考，再作为命令传给 RL 策略执行

### 2.4 数据流全景

```
[观测流]  Unitree LowState (DDS) → G1Env.observe()
            → q/dq/ddq/tau(43) + floating_base(7+6+6) + 手腕位姿(14) + 躯干IMU
            → 下半身 RL 策略观测（86 维单帧 × 6 帧历史 = 516 维）
            → STATE_TOPIC 发布 → 数据导出器（LeRobot 格式）
[指令流]  Teleop/导航/键盘 → G1DecoupledWholeBodyPolicy
            → 合并全身 q → JointSafetyMonitor 安全检查
            → G1Env.queue_action() → 关节序→电机序映射
            → Unitree SDK2 DDS LowCmd（真机）或 MuJoCo 桥（仿真）
```

---

## 3. control/ 核心代码详解

### 3.1 抽象层（`control/base/`）

- `Policy`（`base/policy.py`）：抽象接口 `set_observation()` / `set_goal()` / `get_action()` / `reset()` / `close()`。所有策略（RL、插值、IK 重定向、键盘导航、LeRobot 回放）都实现这个接口，可互换、可组合。
- `Env`（`base/env.py`）：`observe()` / `queue_action()`，带 gymnasium 风格的 observation/action space。

### 3.2 主控制循环（`control/main/teleop/run_g1_control_loop.py`）

50 Hz ROS 2 节点（`ControlLoopConfig.control_frequency = 50`），每周期：

1. （同步模式下）先推进仿真器；
2. `env.observe()` 拿观测，喂给策略；
3. 从 ROS 话题取上半身目标，附带插值垃圾回收时间戳后 `set_goal()`；
4. `wbc_policy.get_action()` → `env.queue_action()`；
5. 发布下半身策略状态、关节安全状态、完整状态（供数据导出）；
6. 处理键盘事件（`]` 激活策略、`o` 停用、`c` 开始/停止采集、`x` 丢弃、`k` 复位仿真）。

内置 `Telemetry` 分段计时，超时打印诊断。安全机制三层：键盘急停（`KeyboardEStop`）、遥操作 1 秒超时注入安全目标、`JointSafetyMonitor` 关节越限检查。

### 3.3 IK 求解器：Pink + Pinocchio + QP

`control/teleop/solver/body/body_ik_solver.py` 的 `BodyIKSolver`：

- 库：**Pink**（`pin-pink`，基于 Pinocchio 的差速 IK 框架）+ **qpsolvers**（优先 `quadprog`，退化为任意可用 QP 求解器）；
- 任务：每个被跟踪的身体环节（手腕等）一个 `FrameTask`（位置+姿态代价），外加一个自定义的 `WeightedPostureTask`（按关节类型加权的姿态正则项，权重表见 `body_ik_solver_settings.py`）；
- 每帧迭代 `num_step_per_frame` 次 `pink.solve_ik()`，积分速度后做关节限位裁剪；
- 手部另有轻量的 `G1GripperInverseKinematicsSolver`（`solver/hand/g1_gripper_ik_solver.py`）：不是 QP，直接从 Manus 手套的拇指/食指/中指指尖位置解析映射到 G1 三指手的 7 个关节。

机器人模型由 `RobotModel`（`control/robot_model/robot_model.py`）封装 Pinocchio URDF（`model_data/g1/g1_29dof_with_hand.urdf`），提供关节组索引、FK 缓存、重力补偿力矩、关节限位裁剪。还有 `ReducedRobotModel`：冻结部分关节做降阶 IK（如只解手臂）。

### 3.4 RL 策略：加载与推理

`G1GearWbcPolicy`（`control/policy/g1_gear_wbc_policy.py`）：

- **模型**：两个 ONNX 策略，用 `onnxruntime` 加载——站立/平衡策略 + 行走策略。发布文件在 `sim2mujoco/resources/robots/g1/policy/`：`GR00T-WholeBodyControl-Balance.onnx` 和 `GR00T-WholeBodyControl-Walk.onnx`（各约 1.9 MB，NVIDIA Open Model License）。
- **双策略切换**：导航速度命令范数 < 0.05 用 Balance，否则用 Walk。
- **观测**（`compute_observation()`，单帧 86 维）：
  - 命令 3（线速度×2+角速度，缩放 [2,2,0.25]）+ 基座高度 1 + 躯干 RPY 命令 3
  - 基座角速度 3（×0.25）+ 投影重力 3
  - 关节位置 29（减默认角度）+ 关节速度 29（×0.05）
  - 上一步动作 15
  - 拼 6 帧历史 → **516 维输入**（与 ONNX 实际输入维度实测一致：`[batch, 516]`）
- **动作**：15 维（腿 12 + 腰 3），`action_scale = 0.25`，叠加到默认站立角度上得到目标关节位置；实测 ONNX 输出 `[batch, 15]`。
- 动作最终是**位置目标**，底层 PD（增益表在 YAML 的 `kps`/`kds`）由宇树电机控制器（真机）或 MuJoCo 桥（仿真）执行。
- 键盘可直接调试 RL 策略：`wasdqe` 速度、`12` 高度、`345678` 躯干姿态、`nm` 步频。

### 3.5 机器人接口层（`control/envs/g1/`）

- `G1Env`（`g1_env.py`）：组合身体+双手+仿真器，负责 actuator 序↔关节序转换、末端位姿观测（Pinocchio FK）、安全监控；
- `G1Body` / `G1ThreeFingerHand`：通过 `unitree_sdk2py` 的 DDS 通道收发 `LowCmd`/`LowState`，**真机与仿真共用同一接口**；
- `sim/unitree_sdk2py_bridge.py`：在 MuJoCo 之上模拟宇树 DDS 协议——这是 sim2real 一致性的核心：控制栈分不清自己在跟真机还是仿真说话。

---

## 4. 与 MuJoCo 仿真的关系（sim2mujoco/）

两层关系：

**第一层：独立的 sim2sim 验证 demo。** `sim2mujoco/scripts/run_mujoco_gear_wbc.py` 是一个自包含脚本（类 `GearWbcController`）：加载 `resources/robots/g1/g1_gear_wbc.xml`，起 `mujoco.viewer`，仿真 dt=0.005（200 Hz），每 4 步跑一次策略（=50 Hz），PD 算力矩直接写 `data.ctrl`。键盘调速度/高度/躯干姿态/步频。`run_mujoco_gear_wbc_gait.py` 是带步态时钟的变体。用途：RL 策略训练完先在纯 MuJoCo 里验证，再进完整控制栈。
⚠️ 注意：`g1_gear_wbc.yaml` 里的 `policy_path` 写的是 `policy/ft92.onnx` / `ft109.onnx`（内部训练编号），仓库实际发布的是 `GR00T-WholeBodyControl-*.onnx`，直接运行需改配置。

**第二层：作为主控制栈的仿真后端。** 控制循环把 `SIMULATOR: mujoco` 时由 `SimulatorFactory` 创建 MuJoCo 仿真（场景见 `control/robot_model/model_data/g1/scene_*.xml`），通过上面的 SDK2 桥伪装成真机。`SIMULATOR: robocasa` 时则走 dexmg 的 RoboCasa 环境（含场景物体、相机渲染）。

---

## 5. 可运行入口

### `scripts/`（顶层编排）

| 脚本 | 演示内容 |
|---|---|
| `scripts/deploy_g1.py` | **一键部署编排器**（类 `G1Deployment`）：用 tmux 起三个窗格——主控制循环+数据采集+遥操作、相机转发、相机预览。真机/仿真通吃 |
| `scripts/leap_tracking_example.py` | LeapMotion 手部追踪最小示例（打印掌心位置） |
| `scripts/run_webcam_recorder.py` | 简单的 webcam 录像工具（数据采集辅助） |

### `control/main/teleop/`（真正的运行入口，`run_*.py`）

| 入口 | 演示内容 |
|---|---|
| `run_g1_control_loop.py` | **主控制循环**：`--interface sim`（默认，MuJoCo 仿真）或 `--interface real`（真机，需静态 IP 192.168.123.222） |
| `run_teleop_policy_loop.py` | 遥操作节点：VR 设备 → Pink IK → ROS 发布上半身关节目标（20 Hz）；支持 `--lerobot_replay_path` 回放数据集代替真人 |
| `run_navigation_policy_loop.py` | 键盘导航节点（10 Hz），发布速度命令 |
| `run_g1_data_exporter.py` | 数据导出节点：订阅状态话题+相机，写 LeRobot/GR00T 格式数据集，带时序一致性监控 |
| `run_sim_loop.py` | 只起仿真器（调试用） |
| `run_sync_sim_data_collection.py` | 同步模式仿真数据采集（控制循环与仿真步进锁定，用于可复现的 CI/数据采集） |
| `playback_sync_sim_data.py` | 回放 hdf5 演示数据（可只显示状态，或通过 WBC 重放动作，可录视频） |
| `run_camera_viewer.py` | 多路相机预览+手动录像 |

典型三件套（官方文档流程）：终端 1 跑控制循环 → 终端 2 跑遥操作（Pico VR）→ `deploy_g1.py` 则把三者+相机一次性编排好。

---

## 6. data/：GR00T 数据管线

- `exporter.py`：`Gr00tDataExporter` **继承 HuggingFace `LeRobotDataset`**，在其上扩展：
  - `meta/modality.json`（GR00T 风格的模态标注，把数据列映射到 GR00T 模型的 state/action/video 语义）；
  - `DataCollectionInfo`（采集元数据：遥操作员、机器 ID、WBC 模型路径等）；
  - 废弃 episode 名单、视频编码（`video_writer.py` 基于 PyAV）。
- `constants.py`：相机分辨率 640×480，数据桶路径 `GearRawDataLeRobotV0`。
- `viz/rerun_viz.py`：用 Rerun 回放/可视化已采集的 episode。

采集到的数据流：主循环每帧把 q、手腕位姿目标（`action.eef`）、基座高度/导航命令、时间戳发到 `STATE_TOPIC` → 导出器拼上相机图像 → LeRobot 格式 episode。这就是 GR00T N1.x VLA 模型的训练数据来源之一。

---

## 7. dexmg/：内嵌的 gr00trobocasa

`dexmg/` 只有一个子包 `gr00trobocasa`，其 `setup.py` 自述为 **"Gr00t RoboCasa for loco-manipulation" v0.2.0**（依赖 `mujoco==3.2.6`、robosuite 生态）。它是 [RoboCasa](https://robocasa.ai/)（厨房操作基准，基于 robosuite/MuJoCo）的 NVIDIA 内部 fork，名字暗示与 DexMimicGen（DexMG）数据生成管线同源。新增内容：

- `robocasa/environments/locomanipulation/`：**移动操作环境**——`LMEnvBase`（场景+物体+成功判据抽象）及其子类 `locomanip_pnp.py`（Pick-and-Place，如 `PnPBottle`、`PnPCube`，与 `model_data/g1/pnp_*.xml` 场景对应）、`locomanip_basic.py`、`locomanip_dc.py`；
- `robocasa/wrappers/ik_wrapper.py`：把 IK 目标位姿注入环境的 wrapper（控制循环的 `set_ik_indicator` 就调它，在仿真里显示遥操作目标）；
- `utils/dexmg_utils.py`：`DexMGConfigHelper`，用多重继承自动给任务生成配置类（子任务分割、成功信号）；
- 场景系统（`utils/scene/`）、纹理随机化、物体库（`models/objects/`）等数据采集基础设施。

**它解决的是"在哪采集仿真演示数据"**：`deploy_g1.py --simulator robocasa --env_name PnPBottle` 时，G1 在 RoboCasa 场景里由真人 VR 遥操作完成移动抓取，整套解耦 WBC 栈原样运行，数据按第 6 节的管线落盘。注意：这里的重点是**移动操作场景**而非灵巧手算法本身。

---

## 8. 与 GEAR-SONIC 的区别与联系

同一仓库里确实是两套 WBC，它们是**前后两代**关系（README 时间线：2025-11-12 首发 Decoupled WBC；2026-02-19 发布 GEAR-SONIC）：

| | **Decoupled WBC**（本目录） | **GEAR-SONIC**（`gear_sonic/` + `gear_sonic_deploy/`） |
|---|---|---|
| 定位 | GR00T N1.5/N1.6 配套的工程化控制器 | 下一代"通用"全身控制器（有白皮书、HF  checkpoint、网页 demo） |
| 控制范式 | **混合**：下半身 RL（两个 ONNX）+ 上半身 IK 插值，分而治之 | **单一端到端 RL 策略**：全身运动追踪（motion tracking），编码器-解码器结构，PPO 在 Isaac Lab 里训练 |
| 训练代码 | 不在仓库里（只发布推理 ONNX） | 完整开源：`gear_sonic/train_agent_trl.py`、数据处理管线、Bones-SEED 数据集 |
| 部署 | Python + ROS 2 + DDS（宇树 SDK） | 额外提供 **C++ 推理栈**（`gear_sonic_deploy/`）面向真机低延迟 |
| 上半身来源 | 遥操作 IK（Pink/QP）或 GR00T 规划器给出的关节目标 | 由策略直接跟踪未来参考运动（约 80–200 ms lookahead） |
| 强项 | 稳定、可解释、好调试，上半身精度由 IK 保证，适合大规模遥操作数据采集 | 全身协调（弯腰拾取、动态动作）、低延迟遥操作、可作为 VLA 的底层执行器 |

联系：
- 同一平台（Unitree G1）、同一团队（NVIDIA GEAR Lab）、同一套遥操作/数据采集理念；`decoupled_wbc` 的下半身策略类干脆就叫 `G1GearWbcPolicy`，配置键叫 `GEAR_WBC_CONFIG`——可以把它理解为 GEAR 系列 WBC 的第一代；
- 两者共享大量基础设施的基因：宇树 DDS 桥、MuJoCo sim2sim、VR 遥操作流、LeRobot 数据格式（SONIC 目录下能看到对应的 `utils/mujoco_sim/`、`utils/teleop/` 等同构模块）；
- 分工上，Decoupled WBC 目前仍是 GR00T N1.5/N1.6 数据收集与部署的文档化路径；GEAR-SONIC 是研究主线和未来方向。

---

## 9. 关键数字速查（讲课用）

| 项目 | 数值 | 出处 |
|---|---|---|
| 机器人 | Unitree G1，29 体自由度（腿 6×2 + 腰 3 + 臂 7×2）+ 三指手 7×2 = 43 | `g1_supplemental_info.py` |
| 下半身 RL 观测 | 单帧 86 = 命令3+高度1+RPY3+角速度3+重力3+q29+dq29+上步动作15；×6 帧历史 = **516** | `g1_gear_wbc_policy.py` + ONNX 实测 |
| 下半身 RL 动作 | **15**（腿 12 + 腰 3），scale 0.25，位置目标+底层 PD | 同上 |
| 上半身 | 17 关节目标（臂 14，可选腰 3），插值限速 | `wbc_policy_factory.py` |
| 主循环 / 遥操作 / 导航 / 仿真 | 50 Hz / 20 Hz / 10 Hz / 200 Hz（dt=0.005，decimation=4） | `configs.py`、各入口、sim2mujoco yaml |
| 策略切换阈值 | 速度命令范数 0.05（Balance ↔ Walk） | `g1_gear_wbc_policy.py` |
| 安全 | 遥操作 1 s 超时、关节限位监控、键盘急停 | `g1_decoupled_whole_body_policy.py`、`joint_safety.py` |

## 10. 已核实的坑（诚实备注）

1. `control/main/teleop/configs/g1_gear_wbc.yaml`（写着 `num_obs: 570`，注释还写 "76 * 6"）是**过期文件**，没有任何代码引用它；实际生效的配置链是 `configs/g1_29dof_gear_wbc.yaml` → `GEAR_WBC_CONFIG` → `sim2mujoco/resources/robots/g1/g1_gear_wbc.yaml`（`num_obs: 516`，与 ONNX 输入一致）。
2. `sim2mujoco` 的 yaml 引用 `ft92.onnx`/`ft109.onnx`，仓库里不存在；发布模型是 `GR00T-WholeBodyControl-Balance.onnx` / `-Walk.onnx`，跑 demo 前需改 `policy_path`。
3. RL 策略代码里有大量被注释掉的步态时钟（gait clock）和躯干观测维度——说明策略观测空间经历过迭代，发布版把这些特征裁掉了。
4. `dexmg/gr00trobocasa` 是 vendored 依赖而非子模块，体积大且与主控栈耦合弱，阅读时可以最后看。
