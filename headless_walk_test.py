"""无头验证 GearWbcController：站立 3 秒 → 前进 5 秒，报告机身高度与位移。"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "decoupled_wbc/sim2mujoco/scripts"))
import mujoco
import torch
from run_mujoco_gear_wbc import GearWbcController

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "decoupled_wbc/sim2mujoco/resources/robots/g1"
)
c = GearWbcController(CONFIG_PATH)
c.config["simulation_duration"] = 9999  # 我们自己控制时长

def step_loop(seconds, cmd=None):
    if cmd is not None:
        c.control_dict["loco_cmd"][0] = cmd
    n_steps = int(seconds / c.model.opt.timestep)
    for _ in range(n_steps):
        leg_tau = c.pd_control(
            c.target_dof_pos, c.data.qpos[7:7 + c.config["num_actions"]],
            c.config["kps"], np.zeros_like(c.config["kps"]),
            c.data.qvel[6:6 + c.config["num_actions"]], c.config["kds"],
        )
        c.data.ctrl[: c.config["num_actions"]] = leg_tau
        if c.n_joints > c.config["num_actions"]:
            arm_tau = c.pd_control(
                np.zeros(c.n_joints - c.config["num_actions"], dtype=np.float32),
                c.data.qpos[7 + c.config["num_actions"]: 7 + c.n_joints],
                np.full(c.n_joints - c.config["num_actions"], 100.0),
                np.zeros(c.n_joints - c.config["num_actions"]),
                c.data.qvel[6 + c.config["num_actions"]: 6 + c.n_joints],
                np.full(c.n_joints - c.config["num_actions"], 0.5),
            )
            c.data.ctrl[c.config["num_actions"]:] = arm_tau
        mujoco.mj_step(c.model, c.data)
        c.counter += 1
        if c.counter % c.config["control_decimation"] == 0:
            single_obs, _ = c.compute_observation(c.data, c.config, c.action, c.control_dict, c.n_joints)
            c.obs_history.append(single_obs)
            for i, hist_obs in enumerate(c.obs_history):
                c.obs[i * c.single_obs_dim: (i + 1) * c.single_obs_dim] = hist_obs
            obs_tensor = torch.from_numpy(c.obs).unsqueeze(0)
            if np.linalg.norm(np.array(c.control_dict["loco_cmd"])) <= 0.05:
                c.action = c.policy(obs_tensor).cpu().detach().numpy().squeeze()
            else:
                c.action = c.walk_policy(obs_tensor).cpu().detach().numpy().squeeze()
            c.target_dof_pos = c.action * c.config["action_scale"] + c.config["default_angles"]

base_xy = lambda: c.data.qpos[0:2].copy()
height = lambda: c.data.qpos[2]

print(f"[t=0]   高度={height():.3f} m  位置={base_xy().round(3)}")
step_loop(3.0, cmd=0.0)          # 站立 3 秒（Balance 策略）
print(f"[t=3s]  站立后: 高度={height():.3f} m  位置={base_xy().round(3)}")
p0 = base_xy()
step_loop(5.0, cmd=0.4)          # 前进指令 0.4 m/s × 5 秒（Walk 策略）
d = np.linalg.norm(base_xy() - p0)
print(f"[t=8s]  前进后: 高度={height():.3f} m  位移={d:.2f} m")
ok = height() > 0.55 and d > 0.8
print("PASS: 机器人站稳并行走" if ok else "FAIL: 摔倒或未行走")
