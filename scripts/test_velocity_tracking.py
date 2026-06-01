import os
import time
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
import tkinter as tk
from threading import Thread

# Helper for CPU-based vector rotation
def rotate_vector_by_quaternion_np(v, q):
    """Rotates a 3D vector v by a unit quaternion q (w, x, y, z) in NumPy."""
    w = q[0]
    xyz = q[1:4]
    xyz_inv = -xyz
    cross1 = np.cross(xyz_inv, v) + w * v
    v_rotated = v + 2.0 * np.cross(xyz_inv, cross1)
    return v_rotated

class Go1InteractiveEnv:
    """A CPU-based Single Environment for Real-Time Go1 Slider Control."""
    def __init__(self, xml_path="../mujoco_menagerie/unitree_go1/scene.xml"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 50Hz control loop (10 physics steps at 0.002s timestep)
        self.frame_skip = 10
        self.dt = self.frame_skip * self.model.opt.timestep 
        
        self.key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.home_qpos = self.model.key_qpos[self.key_id].copy()
        self.home_ctrl = self.model.key_ctrl[self.key_id].copy()
        
        self.action_scale = 0.25
        self.current_action = np.zeros(12, dtype=np.float32)
        
        # Interactive commands [v_x, v_y, w_yaw] modified by the slider GUI
        self.commands = np.zeros(3, dtype=np.float32)
        
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.home_qpos.copy()
        self.data.qvel[:] = 0.0
        self.current_action[:] = 0.0
        
        # Warm up steps
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        return self._get_obs()
        
    def step(self, action):
        self.current_action = action.copy()
        
        # Apply actions
        target_ctrl = self.home_ctrl + (action * self.action_scale)
        self.data.ctrl[:] = target_ctrl
        
        # Simulate steps
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            
        obs = self._get_obs()
        
        # Extract terminal checks
        z_height = self.data.qpos[2]
        base_quat = self.data.qpos[3:7]
        g_global = np.array([0.0, 0.0, -1.0])
        g_local = rotate_vector_by_quaternion_np(g_global, base_quat)
        
        collapsed = z_height < 0.15
        flipped = g_local[2] > -0.5
        terminated = collapsed or flipped
        
        return obs, terminated

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        
        base_quat = qpos[3:7]  # (w, x, y, z)
        v_global = qvel[0:3]
        
        v_local = rotate_vector_by_quaternion_np(v_global, base_quat)
        w_local = qvel[3:6]  # Already local in MuJoCo
        
        g_global = np.array([0.0, 0.0, -1.0])
        g_local = rotate_vector_by_quaternion_np(g_global, base_quat)
        
        joint_pos_error = qpos[7:19] - self.home_qpos[7:19]
        joint_vel = qvel[6:18]
        
        # Build identical 48-dimensional observation vector
        obs = np.concatenate([
            v_local, 
            w_local, 
            g_local, 
            self.commands, 
            joint_pos_error, 
            joint_vel, 
            self.current_action
        ], dtype=np.float32)
        return obs

# Instantiate Environment
env = Go1InteractiveEnv(xml_path="../mujoco_menagerie/unitree_go1/scene.xml")

# --- Tkinter Slider GUI Thread ---
def run_slider_gui():
    root = tk.Tk()
    root.title("Go1 Command Center")
    root.geometry("400x320")
    
    # Thread-safe variables
    vx_var = tk.DoubleVar(value=0.0)
    vy_var = tk.DoubleVar(value=0.0)
    vyaw_var = tk.DoubleVar(value=0.0)
    
    # Update command vector when a slider changes
    def on_slider_change(*args):
        env.commands[0] = vx_var.get()
        env.commands[1] = vy_var.get()
        env.commands[2] = vyaw_var.get()

    # Layout elements
    tk.Label(root, text="Velocity Command Controllers", font=("Helvetica", 14, "bold")).pack(pady=10)
    
    tk.Label(root, text="Forward / Backward Velocity (Vx m/s):", font=("Helvetica", 10)).pack()
    s1 = tk.Scale(root, from_=-1.2, to=1.2, resolution=0.05, orient=tk.HORIZONTAL, length=320, variable=vx_var, command=on_slider_change)
    s1.pack(pady=5)
    
    tk.Label(root, text="Lateral / Sideways Velocity (Vy m/s):", font=("Helvetica", 10)).pack()
    s2 = tk.Scale(root, from_=-0.5, to=0.5, resolution=0.05, orient=tk.HORIZONTAL, length=320, variable=vy_var, command=on_slider_change)
    s2.pack(pady=5)
    
    tk.Label(root, text="Yaw / Turn Rate (Vyaw rad/s):", font=("Helvetica", 10)).pack()
    s3 = tk.Scale(root, from_=-1.5, to=1.5, resolution=0.05, orient=tk.HORIZONTAL, length=320, variable=vyaw_var, command=on_slider_change)
    s3.pack(pady=5)
    
    # Emergency Quick Stop / Stand button
    def reset_sliders():
        vx_var.set(0.0)
        vy_var.set(0.0)
        vyaw_var.set(0.0)
        on_slider_change()
        
    tk.Button(root, text="RESET / STAND STILL", command=reset_sliders, bg="orange", fg="black", font=("Helvetica", 10, "bold")).pack(pady=15)
    
    root.mainloop()

# Launch the Slider GUI on a non-blocking background thread
gui_thread = Thread(target=run_slider_gui, daemon=True)
gui_thread.start()

# --- Main Test Execution Loop ---
def main():
    model_path = "go1_warp_velocity_model.zip"
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found! Check your training output folder.")
        return
        
    print(f"Loading trained PPO policy from: {model_path}")
    model = PPO.load(model_path)
    
    print("\n" + "="*50)
    print("SLIDER TELEOPERATION READY:")
    print("  Drag the GUI sliders to command the robot in real time.")
    print("  The robot will respond smoothly to continuous speed inputs.")
    print("="*50 + "\n")
    
    obs = env.reset()
    
    # Launch passive visualizer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # Predict action deterministically
            action, _ = model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, terminated = env.step(action)
            
            if terminated:
                print("Robot fell or slipped! Resetting...")
                obs = env.reset()
                
            # Sync physics state with the renderer
            viewer.sync()
            
            # Lock loop frequency to 50Hz real-world wall-clock time
            elapsed = time.time() - step_start
            sleep_time = env.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

if __name__ == "__main__":
    main()
