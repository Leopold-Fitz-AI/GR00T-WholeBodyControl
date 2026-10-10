"""可视化 sample_data 里的示例走路动作（第 3 课配套）。"""
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# G1 29 自由度，MuJoCo 执行器顺序（与 gear_sonic/data_process 的 BONES_CSV_JOINT_NAMES 一致）
JOINTS = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]

d = joblib.load("sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl")
m = d["walk_forward_amateur_001__A001"]
fps = m["fps"]
t = np.arange(m["dof"].shape[0]) / fps
root = m["root_trans_offset"]          # (T,3) 根位移
dof = np.rad2deg(m["dof"])             # (T,29) 弧度→角度

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("walk_forward_amateur_001__A001  (G1 retargeted, 30 fps, 1202 frames)", fontsize=13)

# 1) 俯视轨迹（该数据的"前进"轴是 Y）
ax = axes[0, 0]
ax.plot(root[:, 1], root[:, 0], lw=1.2)
ax.scatter(root[0, 1], root[0, 0], c="g", label="start", zorder=3)
ax.scatter(root[-1, 1], root[-1, 0], c="r", label="end", zorder=3)
ax.set_xlabel("Y forward (m)"); ax.set_ylabel("X lateral (m)")
ax.set_title("Root trajectory (top view)"); ax.legend(); ax.grid(alpha=.3)

# 2) 根高度
ax = axes[0, 1]
ax.plot(t, root[:, 2], lw=1.2, color="tab:orange")
ax.set_xlabel("time (s)"); ax.set_ylabel("root height (m)")
ax.set_title("Root height"); ax.grid(alpha=.3)

# 3) 腿部关键关节角（步态周期清晰可见）
ax = axes[1, 0]
for name in ["left_hip_pitch", "right_hip_pitch", "left_knee", "right_knee"]:
    ax.plot(t, dof[:, JOINTS.index(name)], lw=1.0, label=name)
ax.set_xlabel("time (s)"); ax.set_ylabel("angle (deg)")
ax.set_title("Leg joint angles (gait cycle)"); ax.legend(fontsize=8); ax.grid(alpha=.3)

# 4) 踝关节（部署时 kp/kd ×1.5 的那两个电机：索引 4 和 10）
ax = axes[1, 1]
for name in ["left_ankle_pitch", "right_ankle_pitch"]:
    ax.plot(t, dof[:, JOINTS.index(name)], lw=1.0, label=f"{name} (idx {JOINTS.index(name)})")
ax.set_xlabel("time (s)"); ax.set_ylabel("angle (deg)")
ax.set_title("Ankle pitch — motors 4 & 10"); ax.legend(fontsize=8); ax.grid(alpha=.3)

plt.tight_layout()
plt.savefig("walk_motion_overview.png", dpi=110)
print("saved walk_motion_overview.png")
print(f"duration: {t[-1]:.1f}s, forward distance: {root[-1,0]-root[0,0]:.2f} m, "
      f"mean height: {root[:,2].mean():.3f} m")
