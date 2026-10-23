# GR00T-WholeBodyControl 全课程讲义

> 十课系列：由浅入深通读 NVIDIA GR00T 全身控制代码库
> 仓库：`NVlabs/GR00T-WholeBodyControl`（本地路径 `/Users/LeopoldFitz/code/GR00T-WholeBodyControl`）
> 配套材料：`walk_motion_overview.png`（第 3 课）、`visualize_sample_walk.py`、`reports/decoupled_wbc_architecture_report.md`（第 9 课完整报告）

---

## 第 1 课：全局地图

### 1.1 项目定位

NVIDIA 人形机器人（Unitree G1）全身控制（WBC）单仓，目标：让机器人像人一样自然地动（走/跑/蹲/爬/跳/弯腰拾取/双手操作），并支持 VR 遥操作与上层 VLA 大模型（GR00T N 系列）驱动。

### 1.2 三代架构同仓

| 子项目 | 目录 | 范式 | 地位 |
|---|---|---|---|
| Decoupled WBC | `decoupled_wbc/` | 下半身 RL + 上半身 IK 混合 | 第一代（GR00T N1.5/N1.6 配套，仅推理模型） |
| **GEAR-SONIC** | `gear_sonic/`（训练）+ `gear_sonic_deploy/`（C++ 部署） | 单一端到端 RL 运动跟踪 | **当前主力，全开源** |
| MotionBricks | `motionbricks/` | 潜空间生成模型（15,000 FPS 零样本） | 预览版，未来与 SONIC 组成"生成→跟踪"闭环 |

### 1.3 SONIC 核心思想

把**运动跟踪（motion tracking）本身**当作可规模化的训练任务：用 BONES-SEED 数据集（14.2 万条、约 288 小时人体动捕，已重定向到 G1）训练一个统一策略模仿任意参考动作。

### 1.4 关键数字速查

- 机器人：Unitree G1，**29 个驱动关节**（腿 12 + 腰 3 + 臂 14）
- 控制频率：**策略 50 Hz**（物理 200 Hz，decimation 4）
- Latent token：**64 维**（2 tokens × 32 维 FSQ）
- Policy 观测：约 **930 维**（93 维 × 10 帧历史）
- 参考前瞻：默认 10 帧 × 20ms ≈ 200ms（低延迟版 4 帧 ≈ 80ms）
- 训练：PPO，4096 并行环境，24 步 rollout，5 epoch × 4 minibatch

---

## 第 2 课：环境体系与模型下载

### 2.1 五种用例 = 五个独立 venv

| 用例 | 环境 | 安装方式 |
|---|---|---|
| 训练/微调 SONIC | Isaac Lab 的 Python 环境（单独装） | `pip install -e "gear_sonic/[training]"` |
| MuJoCo 仿真 | `.venv_sim` | `bash install_scripts/install_mujoco_sim.sh` |
| VR 遥操作 | `.venv_teleop` | `install_scripts/install_pico.sh` |
| 数据采集 | `.venv_data_collection` | `install_scripts/install_data_collection.sh` |
| 真机部署 | C++ 构建（无需 Python 环境） | 见部署文档 |

设计哲学：**工作站训练 + 机器人部署双端工程**；安装脚本幂等（删旧 venv 重建）；uv 托管 Python 3.10/3.11（不用系统 Python）。

安装脚本六步模式（以 `install_mujoco_sim.sh` 为例）：检测架构 → 确保 uv → 装托管 Python → 重建 venv → `uv pip install -e "gear_sonic[sim]"` → 装 vendored 依赖（`external_dependencies/unitree_sdk2_python`）。

### 2.2 模型下载：`download_from_hf.py`（404 行）

模型权重不在 git 里，从 HuggingFace `nvidia/GEAR-SONIC` 下载：

```bash
python download_from_hf.py                 # 默认：部署 ONNX 三件套 → gear_sonic_deploy/policy/release/
python download_from_hf.py --low-latency   # 低延迟变体（4 帧前瞻 80ms）
python download_from_hf.py --sonic-v1-1    # v1.1（航向归一化 + 手腕增强）
python download_from_hf.py --training      # PyTorch checkpoint + SMPL 数据（~30GB，7 个 tar 分卷）
python download_from_hf.py --sample        # 示例数据（~4MB）
```

**发布清单校验三层防线**（防下载器与发布版本漂移）：

1. `load_release_manifest()`（L147）：下载线上 `config.json`，校验 `schema_version == 1`、`model_id`；
2. `validate_variant_files()`（L179）：下载器硬编码文件清单与 manifest **精确相等比较**；
3. `validate_shared_value()`（L196）：共享文件同理。不一致即报错退出——**fail fast**。

### 2.3 环境检查：`check_environment.py`

Basic（Python 版本/LFS 指针抽检/CUDA/磁盘）+ Training（Isaac Lab、trl、wandb…）+ Deploy（`TensorRT_ROOT`、`libnvinfer.so` 版本解析），失败即 `exit(1)`。

### 2.4 实操记录（本机已完成）

- ✅ `.venv_sim` 已装（uv 托管 Python 3.10 + `gear_sonic[sim]` + unitree SDK）
- ✅ 示例数据已下载到 `sample_data/`（6 个 joblib 压缩 PKL）
- 注意：uv 创建的 venv 默认无 pip，装包用 `uv pip install --python .venv_sim/bin/python <pkg>`
- macOS 本机可跑：MuJoCo 仿真、MotionBricks demo；训练需 Isaac Lab（Linux+GPU），部署需 Jetson/真机

---

## 第 3 课：数据管线——从人体动捕到动作库

### 3.1 三层表征（`sample_data/` 实测）

同一条动作 `walk_forward_amateur_001__A001`（1202 帧 @30fps ≈ 40s）在三个层级各有一份：

```
soma_filtered/   原始人体动捕（SOMA 表征）
  soma_joints (T,26,3) | soma_root_quat (T,4) | soma_transl (T,3) | fps=30

smpl_filtered/   SMPL 标准人体模型（"通用人体语言"）
  pose_aa (T,72) = 24关节×3轴角 | transl (T,3) | smpl_joints (T,24,3)
  fps=50（original_fps=30 重采样）

robot_filtered/  G1 机器人（训练直接用）
  root_trans_offset (T,3) | pose_aa (T,30,3) | dof (T,29) ← MuJoCo 关节序！
  root_rot (T,4) | smpl_joints (T,24,3)（跨域对齐用）| fps=30
```

文件格式：**joblib 压缩**（zlib 魔数 `78 5e`），全仓库统一 `joblib.load` 读取。

### 3.2 关键坑

1. `dof` 是 **MuJoCo 关节序**，Isaac Lab 顺序不同；映射表有三处：`envs/manager_env/robots/g1.py`（训练）、C++ 部署栈、数据转换脚本；
2. `_M` 后缀 = 左右镜像数据增强；
3. 帧率链路：**120（采集）→ 30（存储）→ 50（消费，对齐控制频率）**；
4. SMPL 是 Y-up 坐标系，机器人世界 Z-up，进机器人前 +90° X 旋转。

### 3.3 转换脚本

`gear_sonic/data_process/convert_soma_csv_to_motion_lib.py`（584 行）：BONES-SEED CSV（120fps，度/cm）→ 单位换算 → 欧拉→轴角 → 按 `BONES_CSV_JOINT_NAMES` 对齐 MJCF 执行器序 → 降采样 30fps → joblib pkl。配套 `filter_and_copy_bones_data.py` 过滤低质动作。

### 3.4 训练消费：MotionLib

`utils/motion_lib/motion_lib_robot.py`（`MotionLibRobot`）：加载 pkl → `Humanoid_Batch`（MJCF FK 批量解析器）展开全身 30 刚体状态 → 重采样 50fps。**自适应采样**（PHC 血统）：按失败率动态调整采样概率，失败多的动作更常被采样。

### 3.5 可视化

`visualize_sample_walk.py` 生成 `walk_motion_overview.png`：俯视轨迹（前 10s 站立，行走约 8m，横向 ±0.1m 重心摆动）、根高度（0.79→0.75m）、膝关节步态周期（摆动相屈曲 ~80°）、踝 pitch（电机索引 4/10，即部署增益缩放的目标）。

---

## 第 4 课：RL 环境——Isaac Lab Manager 架构

### 4.1 Manager-Based 心智模型

**环境 = 场景 + 一组管理器，每个管理器负责 MDP 一个侧面，每个侧面由若干 term（YAML 小文件）装配。** 配置目录 `gear_sonic/config/manager_env/` 与 Isaac Lab manager 一一对应：`actions/`、`observations/{policy,critic,tokenizer}/`、`rewards/`、`terminations/`、`commands/`、`events/`、`curriculum/`。每个 term 的 `_target_` 指向 `envs/manager_env/mdp/*.py` 中的函数——**配置即装配**。

### 4.2 动作：29 维关节位置目标

`actions/terms/joint_pos.yaml` → `JointPositionActionCfg`：

```
q_target = q_default + action × action_scale
action_scale = 0.25 × effort_limit / stiffness  （每关节不同）
```

**训练侧与部署侧的动作换算必须逐字节一致**——全项目最重要的 sim2real 不变量。

### 4.3 三组观测，各伺候一个网络

| 组 | 喂给谁 | 内容 |
|---|---|---|
| **policy**（`observations/policy/local_dir_hist.yaml`） | 解码器（部署可用） | 重力方向(3)+基座角速度(3)+关节位置(29)+关节速度(29)+上帧动作(29) = 93 维 × 10 帧历史 ≈ **930 维**，带噪声 corruption |
| **critic**（`privileged_mf_hist.yaml`） | 价值网络（仅训练） | policy 全部 + 特权信息（未来参考帧、全身 14 刚体位置/朝向、线速度） |
| **tokenizer**（`unitoken_all_noz.yaml`） | 编码器 | 三种模态未来轨迹（g1 / teleop / smpl）+ `encoder_index` 选择编码器 |

设计要点：**非对称 Actor-Critic**（训练开卷、部署闭卷）；历史帧堆叠给无记忆的 MLP 提供时序；观测维度不写死，`env.reset()` 后内省回填（`get_group_term_obs_shape()`）。

### 4.4 Command：`TrackingCommand`

`envs/manager_env/mdp/commands.py`（4310 行）：持有 MotionLibRobot，episode reset 采样 motion_id+起始帧，每步推进时间游标产出当前帧+未来参考帧；每 episode 按 `{g1:1, teleop:1, smpl:1}` 抽命令模态；增强：上半身动作拼接（50%）、冻结帧、行走类只增强上半身。约 60 个 property 访问器（命名规范 `{量}[_multi_future]_[w|l|b]`）。

### 4.5 奖励与终止

`rewards/tracking/base_5point_local_feet_acc.yaml`，跟踪项全是高斯核 `w·exp(-‖err‖²/σ²)`：

| 项 | 权重 | σ |
|---|---|---|
| anchor 位置/朝向 | 0.5/0.5 | 0.3/0.4 |
| 相对刚体位置/朝向 | 1.0/1.0 | 0.3/0.4 |
| 刚体线速度/角速度 | 1.0/1.0 | 1.0/3.14 |
| **5 点局部跟踪（双腕+头+双脚）** | **2.0** | **0.1**（最严） |

正则负奖励：action_rate、关节限位、多余接触、抖动、脚部加速度。终止：锚点偏差 >0.15m / 朝向 >0.2 / 脚 >0.2（自适应累计误差机制）。域随机化（`events/tracking/level0_4.yaml`）：摩擦、质心、每 4–6s 推搡 ±0.5m/s、手腕/躯干质量 ×0.8–2.5。

### 4.6 时间结构

物理 200 Hz（sim_dt=0.005）→ decimation 4 → 策略 50 Hz；episode 10s。

---

## 第 5 课：Universal Token 网络架构 ★核心课

代码：`gear_sonic/trl/modules/universal_token_modules.py`（1256 行）；配置：`config/actor_critic/universal_token/all_mlp_v1.yaml`。

### 5.1 架构总览

```
tokenizer_obs（三种模态未来轨迹）
  ├─► g1 编码器     MLP [2048,1024,512,512] ─┐
  ├─► teleop 编码器  MLP（同构）              ├─► FSQ 量化 ─► 2×32=64 维 token
  └─► smpl 编码器    MLP（同构）              ─┘        │
                                                        ▼
  g1_dyn 解码器（主策略）: token(64) ⊕ proprioception(930) → MLP [2048,2048,1024,1024,512,512] → 29 维动作
  g1_kin 解码器（训练专用）: token → 重构未来参考轨迹（辅助损失）
```

关键配置：`num_fsq_levels: 32`、`max_num_tokens: 2` → token 64 维。

### 5.2 FSQ（Finite Scalar Quantization）

每维值域均匀切 32 级、四舍五入——无码本、不坍缩、天然有界（理论码本 32³²）。对比 VQVAE 学码本（第 10 课）。有界网格潜空间对下游 VLA 学习输出 token 非常友好。

### 5.3 三编码器输入分工（配置原文）

- `g1`: `command_multi_future_nonflat`(10,58)、`motion_anchor_ori_b_mf_nonflat`(10,6)、`command_z_multi_future_nonflat`
- `teleop`: 下肢 12 关节未来轨迹、`vr_3point_local_target`(9)、`vr_3point_local_orn_target`(18)、`motion_anchor_ori_b`、`command_z`
- `smpl`: `smpl_joints_multi_future_nonflat`(10,72)、`smpl_root_ori_b_multi_future`

VR 三点驱动 29 关节是欠定问题——补全能力来自潜空间跨模态对齐。

### 5.4 辅助损失（`aux_losses/universal_token/g1_recon_and_all_latent.yaml`）

| 损失 | 系数 | 作用 |
|---|---|---|
| `g1_recon` | 0.01 | token 重构未来轨迹（保证 token 含信息） |
| `g1_smpl_latent` / `g1_teleop_latent` / `teleop_smpl_latent` | 各 1.0 | 同一动作跨编码器 token 两两对齐（MSE） |
| `reencoded_smpl_g1_latent` | 1.0 | 循环一致性：smpl→token→重构→g1 再编码≈原 token |

### 5.5 三种推理模式（VLA 分层接口）

`residual`（策略输出对编码 token 的残差）/ `direct_latent`（直接输出完整 token，跳过编码器——VLA 部署形态）/ `mixed`（teacher/student 混合蒸馏）。

### 5.6 代码方法地图

`__init__`(L69) → `parse_tokenizer_obs`(L478) → `create_encoder_masks`(L506) → `_encode_single`(L586)/`encode`(L662) → `assemble_all_tokens`(L552) → `decode`(L702) → `forward`(L751)；部署路径 `forward_with_external_tokens`(L1113)；ONNX 导出 `get_all_example_input`(L1209)。

---

## 第 6 课：PPO 训练循环——`train_agent_trl.py`（481 行）

### 6.0 序幕坑（L16–26）

本地包 `gear_sonic/trl/` 与 HuggingFace `trl` **同名撞车**：脚本开头手动修 `sys.path`，否则 `import trl` 静默解析到本地目录。

### 6.1 七步启动流程

1. **Hydra 配置**（L155）：`config/base.yaml` + `+exp=manager/universal_token/all_modes/sonic_release` 叠加；
2. **断点续训**（L73–91）：glob `{实验目录}-*/last.pt`，wandb run id 存 `meta.yaml`；
3. **HF 参数解析**（L159–172）：`HfArgumentParser` 读 `config/algo/trl/ppo.yaml`（转发壳，真超参在 `ppo_im_phc.yaml`）；
4. **Accelerate/DDP**（L179–198）：多卡 seed 按 rank 偏移；
5. **Isaac Sim 启动**（L228–279）：必须在 torch 大量使用 CUDA 之前；多进程用 `/tmp/isaaclab_app_launcher.lock` 文件锁；
6. **环境创建 + 观测内省**（L321, L358–379）：`ManagerBasedRLEnv` + `ManagerEnvWrapper`（2106 行胶水）；reset 后回填观测维度；
7. **模型 + Trainer**（L354–470）：Actor（UniversalTokenModule）/ Critic / 可选 ref_model(DAgger) / disc_model(AMP)；`materialize_lazy_params`；`pretrained_model.module_mapping` 子模块级预训练加载；`TRLAuxLossPPOTrainer`。

### 6.2 PPO 超参（`config/algo/ppo_im_phc.yaml`）——"精修配方"

| 参数 | 值 | 点评 |
|---|---|---|
| rollout | 24 步/env（sonic_release 覆盖 32） | |
| epoch × minibatch | 5 × 4 | |
| clip / γ / λ | 0.2 / 0.99 / 0.95 | 经典 |
| actor lr / critic lr | **2e-5** / 1e-3 | critic 快 50 倍 |
| `schedule: adaptive`, `desired_kl: 0.01` | lr ∈ [1e-5, 2e-4] | KL 自适应学习率 |
| `init_noise_std` | **0.05** | 模仿学习只需细调 |
| iterations / save_interval | 100000 / 500 | |

### 6.3 主循环

rollout(4096 env × 24 步) → GAE → PPO(5×4) + latent 辅助损失 → 跨 GPU 同步均值/采样权重 → callbacks（model_save / wandb / im_resample / im_eval）。

继承链：HF `PPOTrainer` → `TRLPPOTrainer`（RL 环境适配）→ `TRLAuxLossPPOTrainer`（辅助损失）。

### 6.4 隐藏模式

`replay=True` 回放动作库（检查数据质量）；`vplanner_replay=True` 回放 V-planner 加噪声测鲁棒性。

---

## 第 7 课：C++ 部署栈 ★

`gear_sonic_deploy/`，唯一产物：可执行文件 `g1_deploy_onnx_ref`（主文件 4535 行）。

### 7.1 四频率线程模型（全部 SCHED_FIFO 实时调度、绑核）

| 线程 | 频率 | 职责 |
|---|---|---|
| Input | 100 Hz | 输入轮询（键盘/手柄/ZMQ/ROS2） |
| **Control** | **50 Hz** | 主控制环 |
| Planner | 10 Hz | 运动规划 |
| CommandWriter | 500 Hz | 最新 MotorCommand → CRC → DDS |

+ DDS 回调：LowStateHandler 500Hz（状态+错误监控）、imuTorsoHandler。线程间 `DataBuffer<T>` 只留最新值。**慢线程决策、快线程复读**。

### 7.2 50 Hz 主循环 10 步流水线（`Control()`, L3834）

采集状态（关节重排 mujoco_to_isaaclab；速度 >35 rad/s 停机）→ 快照输入 → 组装观测（YAML 声明式 gather 注册表）→ 日志 → TensorRT 推理 → 灵巧手 → 广播状态（ZMQ 5557）→ 录制 → 参考帧推进（planner 新轨迹 8 帧线性融合）→ 每秒打印 µs 级耗时。

状态机：INIT（3s 插值到站姿）→ WAIT_FOR_CONTROL → CONTROL。

### 7.3 TensorRT 推理

ONNX 仅交付格式：ONNX+GPU型号+精度 → SHA256 → 编译缓存 `.trt` 引擎。三引擎：`EncoderEngine`（观测→64 维 token，支持 mode g1/teleop/smpl 或外部 token 模式 `-1`）、`PolicyEngine`（token+本体感觉→29 维动作）、`LocalMotionPlannerTensorRT`。**CUDA Graph 捕获重放**，Jetson Orin 上 planner 1–2ms。

### 7.4 ZMQ 1280 字节自描述协议

`zmq_packed_message_subscriber.hpp` L99 `HEADER_SIZE = 1280`：

```
[topic 前缀] [1280B JSON header (\0填充)] [二进制字段拼接]
header = {v, endian, count, fields:[{name, dtype, shape}]}
```

Python 改数据结构**不用重编译 C++**；单帧 + `conflate` 保证最新；endian 字段支持 x86/ARM 混合组网。zmq_manager 订阅三 topic：`command` / `planner`（1s 无消息回 IDLE）/ `pose`。

### 7.5 硬件接口与增益缩放

宇树 DDS：`rt/lowstate`（500Hz）/ `rt/lowcmd`；启动先 `MotionSwitcherClient` 释放原厂运控。基准 kp/kd 来自电枢二阶临界阻尼模型（ω=10Hz, ζ=2.0）。`--motor-kp-scale 4,10=1.5`（左右踝 pitch）：解析严格校验，系数钳制 ≤4.0（防手滑 1.5→15）。

### 7.6 七层安全

① CRC + 失联停机 ② 电机故障码/速度硬限/温度迟滞 ③ token 流超时报警 ④ TTS 语音报警（边沿触发）⑤ 参数钳制 ⑥ Idle 重适应（目标向现实 2%/tick 让步再恢复）⑦ 优雅停机（kp=0, kd=8 软着陆）。

---

## 第 8 课：VR 遥操作与 VLA 闭环

### 8.1 协议拓扑

| 端口 | 方向 | 内容 |
|---|---|---|
| 5550 | REQ/REP | GR00T PolicyServer（VLA） |
| 5555 | PUB/SUB | 相机图传 |
| **5556** | PUB/SUB | **动作/命令流（遥操作 & VLA 统一入口）** |
| 5557 | PUB/SUB | 机器人状态回传 |
| 5580 | PUB/SUB | 键盘控制 |

### 8.2 VR 遥操作（`scripts/pico_manager_thread_server.py`, 2311 行）

输入源 `xrt`（PICO，`PicoReader` L801）或 `isaac-teleop`（进程内 CloudXR，仅 G1+Thor）。核心：`ThreePointPose`(L942) 三点标定坐标变换、`PoseStreamer`(L1238) 50Hz 主流循环、`YawAccumulator`(L520) 摇杆转身、手部 `g1_gripper_ik_solver`。三点→全身的补全能力来自训练时潜空间对齐（第 5 课回报）。`FeedbackReader` 订阅 5557 形成闭环。

### 8.3 VLA 闭环（`scripts/run_vla_inference.py`）

每 50Hz：SUB 5557 状态 + 相机图像 → GR00T PolicyClient(5550) 推理 → `pack_latent_action_message`(L137) 打包 64 维 latent + 手部关节 → PUB 5556。大模型输出 token 而非关节角——**分层智能：大模型管"做什么"，控制器管"怎么稳住"**。

### 8.4 编排与数据飞轮

`launch_inference.py`：tmux 四 pane（C++ 控制环 / 键盘 / VLA / 数据导出），`--sim` 加 MuJoCo 窗口。相机栈 `gear_sonic/camera/`（多相机线程 + 5555 PUB + systemd 自启）。数据飞轮：遥操作采数据（LeRobot 格式）→ 微调 GR00T → VLA 部署 → 失败案例补录。

---

## 第 9 课：Decoupled WBC——上一代架构对比

完整报告：`reports/decoupled_wbc_architecture_report.md`。

### 9.1 定义与汇合点

下半身（腿+腰 15 DOF）RL 闭环 + 上半身（臂 14 DOF）IK 限速插值，汇合于 `G1DecoupledWholeBodyPolicy`，腰部冲突时**下半身优先**。

### 9.2 三个设计亮点

① **IK 不在主控制循环**：IK 跑在独立遥操作进程（Pink+QP, 20Hz），经 ROS2 话题发关节目标，主循环只做插值——隔离不确定性；② **解耦不孤立**：Pinocchio FK 算躯干姿态回馈下半身 RL；③ `unitree_sdk2py_bridge.py` 在 MuJoCo 上模拟 DDS，sim2real 一致。

### 9.3 新旧对比

| | Decoupled WBC | GEAR-SONIC |
|---|---|---|
| RL 观测 | 86×6=516 维 | 93×10≈930 维 |
| RL 动作 | 15 维（腿腰） | 29 维（全身） |
| 策略数 | 2（Balance/Walk 按速度 0.05 切换） | 1（统一） |
| 上半身 | IK 插值 | 策略跟踪 |
| 栈 | Python+ROS2 | C+++TensorRT |
| 训练 | 不开源 | 全开源 |

### 9.4 为什么被取代

双策略切换有边界裂缝；IK 无法全身协调（弯腰拾取要腰腿臂配合）；Python/ROS2 延迟到顶；VLA 接口维度太高。但 Decoupled 稳定可解释，仍是 N1.5/N1.6 数据采集文档化路径。

### 9.5 坑

`configs/g1_gear_wbc.yaml` 是过期僵尸配置（num_obs 570≠实际 516）；sim2mujoco yaml 指向不存在的 `ft92.onnx`。

---

## 第 10 课：MotionBricks——生成式运动模型

### 10.1 运动表征（`motionbricks/docs/motion_representation.md`）

**root/body 分离**。三套表征无损互转：`GlobalRootGlobalJoints`(414 维，root model 用) / `LocalRootGlobalJoints`(413，pose/tokenizer 用) / `DualRootGlobalJoints`(418)。body 409 维 = ric_data 99 + global_rot 204（6D 旋转）+ local_vel 102 + foot_contacts 4。分离的意义：root 管导航、pose 管风格，独立可控。

### 10.2 三件套

① **VQVAE**（动作分词器，多头 EMA 码本 512×4——对比 SONIC 的 FSQ）；② **Root model**（Transformer，直接回归全局根轨迹，不 token 化）；③ **Pose model**（Transformer，masked token 预测，embedding 用 VQVAE 码本初始化）。

### 10.3 交互 Demo（`scripts/interactive_demo_g1.py`，173 行）

每帧：键盘 → 控制信号 → root model 出轨迹 → pose model 出 token → VQVAE 解码 → MuJoCo qpos → 渲染。checkpoint 自包含（权重+hparams+skeleton+归一化 stats）。

### 10.4 与 SONIC 的分工

| | SONIC | MotionBricks |
|---|---|---|
| 范式 | 跟踪（RL） | 生成（VQVAE+Transformer） |
| 角色 | 执行层（脊髓） | 规划/内容层（脑） |
| 数据 | BONES-SEED | BONES-SEED（同源） |

未来闭环：MotionBricks 生成意图级动作 → SONIC 物理稳定执行。

---

## 总结：只记三件事

1. **Universal Token 是灵魂**——64 维潜空间统一遥操作、动捕跟踪、VLA 三种意图来源；
2. **sim2real 一致性是不变量**——关节映射表、动作缩放公式在训练/部署侧逐字节对齐；
3. **机器人软件工程文化**——fail fast、频率分层、七层安全，失控的代价是物理损坏。

## 延伸阅读路径

- 想读代码：按第 5.6 节方法地图从 `universal_token_modules.py` 入手；
- 想跑起来：`.venv_sim` + `python gear_sonic/scripts/run_sim_loop.py`（需先 `download_from_hf.py` 下载 ONNX）；
- 想看数据：`.venv_sim/bin/python visualize_sample_walk.py`；
- 想深入旧架构：`reports/decoupled_wbc_architecture_report.md`。
