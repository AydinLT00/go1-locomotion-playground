import os
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import warp as wp
import mujoco
import mujoco_warp as mjw

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

# 1. Initialize Warp on GPU
wp.init()
wp.config.quiet = True
device_str = "cuda:0"
wp.set_device(device_str)
torch_device = torch.device(device_str)

def rotate_vector_by_quaternion(v, q):
    """Rotates batch of 3D vectors v by unit quaternions q (w, x, y, z)."""
    w = q[:, 0:1]
    xyz = q[:, 1:4]
    xyz_inv = -xyz
    cross1 = torch.cross(xyz_inv, v, dim=-1) + w * v
    v_rotated = v + 2.0 * torch.cross(xyz_inv, cross1, dim=-1)
    return v_rotated

class Go1WarpVecEnv(VecEnv):
    """GPU-Parallelized MuJoCo Env for Emergent Velocity Command Tracking."""
    
    def __init__(self, num_envs=2048, xml_path="../mujoco_menagerie/unitree_go1/scene.xml", max_steps=1000):
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.device = torch_device
        self.frame_skip = 10  # 500Hz physics -> 50Hz control loop

        # 2. Load and Compile Model
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.dt = self.frame_skip * self.mj_model.opt.timestep  # 0.02s

        # 3. Observation Space (Size 48):
        # - v_local (3), w_local (3), g_local (3), commands (3), 
        # - joint_pos_error (12), joint_vel (12), last_actions (12)
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(48,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        super().__init__(num_envs, observation_space, action_space)

        # 4. Bind Parallel Data on GPU Device
        self.mjw_model = mjw.put_model(self.mj_model)
        self.mjw_data = mjw.put_data(self.mj_model, self.mj_data, nworld=self.num_envs)

        # Cache initial states
        self.trunk_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.key_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        
        home_qpos_np = self.mj_model.key_qpos[self.key_id].copy()
        home_ctrl_np = self.mj_model.key_ctrl[self.key_id].copy()
        self.home_qpos_torch = torch.tensor(home_qpos_np, device=self.device, dtype=torch.float32)
        self.home_ctrl_torch = torch.tensor(home_ctrl_np, device=self.device, dtype=torch.float32)

        # Zero-Copy PyTorch Tensors
        self.qpos_tensor = wp.to_torch(self.mjw_data.qpos)
        self.qvel_tensor = wp.to_torch(self.mjw_data.qvel)
        self.ctrl_tensor = wp.to_torch(self.mjw_data.ctrl)

        # Command & Tracking States
        self.commands = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.command_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.resample_interval_steps = 300  # Resample target commands every 6 seconds (300 steps)
        
        # Environment Trackers
        self.episode_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.current_action_tensor = torch.zeros((self.num_envs, 12), device=self.device, dtype=torch.float32)
        self.last_action_tensor = torch.zeros((self.num_envs, 12), device=self.device, dtype=torch.float32)
        self.last_dof_vel = torch.zeros((self.num_envs, 12), device=self.device, dtype=torch.float32)
        
        # Action scale set to match Go1RoughCfg
        self.action_scale = 0.25 

        # Initial Command Generation
        self._resample_commands(torch.arange(self.num_envs, device=self.device))

        # Capture Physics Step Loop with CUDA Graphs
        mjw.step(self.mjw_model, self.mjw_data)
        
        print("Capturing CUDA simulation graph...")
        with wp.ScopedCapture() as capture:
            for _ in range(self.frame_skip):
                mjw.step(self.mjw_model, self.mjw_data)
        self.advance_sim_graph = capture.graph
        print("CUDA Graph Capture Complete!")

    def _resample_commands(self, env_ids):
        """Generates random tracking commands matching Go1RoughCfg distributions."""
        # x_vel target: [-1.0, 1.0] m/s
        self.commands[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * 2.0 - 1.0
        # y_vel target: [-0.4, 0.4] m/s
        self.commands[env_ids, 1] = torch.rand(len(env_ids), device=self.device) * 0.8 - 0.4
        # yaw target: [-1.0, 1.0] rad/s
        self.commands[env_ids, 2] = torch.rand(len(env_ids), device=self.device) * 2.0 - 1.0
        self.command_steps[env_ids] = 0

    def step_async(self, actions):
        action_tensor = torch.tensor(actions, device=self.device, dtype=torch.float32)
        target_angles = self.home_ctrl_torch + (action_tensor * self.action_scale)
        self.ctrl_tensor[:, :] = target_angles
        
        self.last_action_tensor = self.current_action_tensor.clone()
        self.current_action_tensor = action_tensor.clone()

    def step_wait(self):
        # Store joint velocities from the previous step to compute dof_acc rewards
        self.last_dof_vel = self.qvel_tensor[:, 6:18].clone()

        wp.capture_launch(self.advance_sim_graph)
        wp.synchronize_device() 

        qpos = self.qpos_tensor
        qvel = self.qvel_tensor
        
        # State extraction
        base_quat = qpos[:, 3:7]  # (w, x, y, z)
        v_global = qvel[:, 0:3]
        
        # Rotational velocities of a free joint are ALREADY local in MuJoCo!
        v_local = rotate_vector_by_quaternion(v_global, base_quat)
        w_local = qvel[:, 3:6]  # FIXED: Removed the secondary quaternion rotation
        
        g_global = torch.tensor([[0.0, 0.0, -1.0]], device=self.device).expand(self.num_envs, 3)
        g_local = rotate_vector_by_quaternion(g_global, base_quat)
        
        z_height = qpos[:, 2]
        joint_vel = qvel[:, 6:18]
        
        self.episode_steps += 1
        self.command_steps += 1

        # Periodic Command Resampling
        resample_mask = self.command_steps >= self.resample_interval_steps
        if resample_mask.any():
            resample_ids = torch.where(resample_mask)[0]
            self._resample_commands(resample_ids)

        # ----------------------------------------------------
        # REWARD FUNCTIONS (Derived directly from legged_gym)
        # ----------------------------------------------------
        
        # 1. Linear velocity tracking (xy-plane)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - v_local[:, :2]), dim=-1)
        r_tracking_lin_vel = torch.exp(-lin_vel_error / 0.25)
        
        # 2. Angular velocity tracking (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - w_local[:, 2])
        r_tracking_ang_vel = torch.exp(-ang_vel_error / 0.25)
        
        # 3. Penalize vertical linear velocity (z)
        r_lin_vel_z = torch.square(v_local[:, 2])
        
        # 4. Penalize non-yaw angular velocities (roll/pitch axes)
        r_ang_vel_xy = torch.sum(torch.square(w_local[:, :2]), dim=-1)
        
        # 5. Penalize non-flat orientation (projected gravity vector roll/pitch components)
        r_orientation = torch.sum(torch.square(g_local[:, :2]), dim=-1)
        
        # 6. Proxy actuator torques penalty based on PD Control:
        # u = Kp * (target - position) - Kd * velocity (values match your XML class rules)
        target_angles = self.home_ctrl_torch + (self.current_action_tensor * self.action_scale)
        torques = 100.0 * (target_angles - qpos[:, 7:19]) - 2.0 * joint_vel
        r_torques = torch.sum(torch.square(torques), dim=-1)
        
        # 7. Penalize joint velocities
        r_dof_vel = torch.sum(torch.square(joint_vel), dim=-1)
        
        # 8. Penalize joint accelerations (calculated using the stored velocities)
        r_dof_acc = torch.sum(torch.square((joint_vel - self.last_dof_vel) / self.dt), dim=-1)
        
        # 9. Penalize rapid changes in action commands (Action Rate)
        r_action_rate = torch.sum(torch.square(self.current_action_tensor - self.last_action_tensor), dim=-1)

        # Terminations & numerical exception guards
        collapsed = z_height < 0.15
        flipped = g_local[:, 2] > -0.5
        is_exploded = torch.isnan(z_height) | torch.isinf(z_height)
        
        terminated = collapsed | flipped | is_exploded
        truncated = self.episode_steps >= self.max_steps
        needs_reset = terminated | truncated

        r_termination = -10.0 * terminated.float()

        # Weighted combination of rewards from Go1RoughCfg scale rules
        rewards = (
            1.0 * r_tracking_lin_vel +
            0.5 * r_tracking_ang_vel +
            -2.0 * r_lin_vel_z +
            -0.05 * r_ang_vel_xy +
            -2.0 * r_orientation +
            -0.0002 * r_torques +
            -0.0005 * r_dof_vel +
            -2.5e-7 * r_dof_acc +
            -0.01 * r_action_rate +
            r_termination
        )

        # Build clean observation structure
        joint_pos_error = qpos[:, 7:19] - self.home_qpos_torch[7:19]
        obs = torch.cat([
            v_local, w_local, g_local, 
            self.commands, 
            joint_pos_error, joint_vel, 
            self.current_action_tensor
        ], dim=-1)

        # Construct Stable-Baselines3 "infos" dictionary
        infos = [{} for _ in range(self.num_envs)]
        
        # In-place Resets
        if needs_reset.any():
            reset_indices = torch.where(needs_reset)[0]
            
            # Populate terminal observations for the value function critic bootstrapping
            terminal_obs = obs.clone()
            for idx in reset_indices:
                infos[idx]["terminal_observation"] = terminal_obs[idx].cpu().numpy()
                if self.episode_steps[idx] >= self.max_steps:
                    infos[idx]["TimeLimit.truncated"] = True

            # Perform resets
            self.qpos_tensor[reset_indices] = self.home_qpos_torch.clone()
            noise_pos = (torch.rand((len(reset_indices), 12), device=self.device) - 0.5) * 0.05
            self.qpos_tensor[reset_indices, 7:19] += noise_pos
            
            self.qvel_tensor[reset_indices] = (torch.rand((len(reset_indices), 18), device=self.device) - 0.5) * 0.02
            
            self.episode_steps[reset_indices] = 0
            self.current_action_tensor[reset_indices] = 0.0
            self.last_action_tensor[reset_indices] = 0.0
            self.last_dof_vel[reset_indices] = 0.0
            
            # Assign new tracking targets for the reset agents
            self._resample_commands(reset_indices)

            # Re-evaluate observation states for the newly reset indices
            qpos_reset = self.qpos_tensor
            qvel_reset = self.qvel_tensor
            
            base_quat_reset = qpos_reset[reset_indices, 3:7]
            v_global_reset = qvel_reset[reset_indices, 0:3]
            
            v_local[reset_indices] = rotate_vector_by_quaternion(v_global_reset, base_quat_reset)
            w_local[reset_indices] = qvel_reset[reset_indices, 3:6]  # Fixed: direct local assign
            
            g_global_reset = torch.tensor([[0.0, 0.0, -1.0]], device=self.device).expand(len(reset_indices), 3)
            g_local[reset_indices] = rotate_vector_by_quaternion(g_global_reset, base_quat_reset)
            
            rewards[reset_indices] = 0.0

            # Reassemble observation vector incorporating the newly reset configurations
            joint_pos_error = qpos_reset[:, 7:19] - self.home_qpos_torch[7:19]
            obs = torch.cat([
                v_local, w_local, g_local, 
                self.commands, 
                joint_pos_error, qvel_reset[:, 6:18], 
                self.current_action_tensor
            ], dim=-1)

        return (
            obs.cpu().numpy(),
            rewards.cpu().numpy(),
            needs_reset.cpu().numpy(),
            infos
        )

    def reset(self):
        mjw.reset_data(self.mjw_model, self.mjw_data)
        self.qpos_tensor[:, :] = self.home_qpos_torch.clone()
        self.qvel_tensor[:, :] = 0.0
        
        self.episode_steps[:] = 0
        self.current_action_tensor[:, :] = 0.0
        self.last_action_tensor[:, :] = 0.0
        self.last_dof_vel[:, :] = 0.0
        
        self._resample_commands(torch.arange(self.num_envs, device=self.device))
        
        mjw.step(self.mjw_model, self.mjw_data)
        obs, _, _, _ = self.step_wait()
        return obs

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        if hasattr(self, attr_name):
            val = getattr(self, attr_name)
            num = self.num_envs if indices is None else len(indices)
            return [val] * num
        raise AttributeError(f"Go1WarpVecEnv has no attribute '{attr_name}'")

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            res = method(*method_args, **method_kwargs)
            num = self.num_envs if indices is None else len(indices)
            return [res] * num
        raise AttributeError(f"Go1WarpVecEnv has no method '{method_name}'")

    def env_is_wrapped(self, wrapper_class, indices=None):
        num = self.num_envs if indices is None else len(indices)
        return [False] * num


if __name__ == "__main__":
    log_dir = "./tb_logs/"
    os.makedirs(log_dir, exist_ok=True)

    num_parallel_envs = 2048
    print(f"Initializing {num_parallel_envs} parallel environments on GPU...")
    
    raw_env = Go1WarpVecEnv(num_envs=num_parallel_envs)
    env = VecMonitor(raw_env)

    # Optimized PPO configuration matching the legged_gym training schedule
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=32,            # Much faster policy updates (close to num_steps_per_env=24)
        batch_size=4096,       
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.003,        # Matches go1 entropy requirements
        verbose=1,
        tensorboard_log=log_dir
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=200_000, 
        save_path="./models_warp/",
        name_prefix="go1_warp_ppo_velocity"
    )

    print("Starting Training! Track performance using TensorBoard.")
    model.learn(
        total_timesteps=20_000_000,
        callback=checkpoint_callback,
        tb_log_name="go1_warp_velocity_tracking"
    )

    model.save("go1_warp_velocity_model")
    print("Training finished on GPU. saved on go1_warp_velocity_model")
