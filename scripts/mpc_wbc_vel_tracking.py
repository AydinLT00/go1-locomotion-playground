import time
import threading
import tkinter as tk
import mujoco
import mujoco.viewer
import numpy as np
import qpsolvers  # Fast QP solver interface
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION & CONSTANTS ---
GAIT_PHASE_STANCE = 1
GAIT_PHASE_SWING = 0

class Go1Controller:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # Robot physical parameters
        self.mass = np.sum(model.body_mass)  
        self.I_body = np.diag([0.07, 0.063, 0.016]) 
        
        # Fast MPC Parameters
        self.N_horizon = 10  # Increased horizon           
        self.dt_mpc = 0.01                    
        self.state_dim = 12
        self.input_dim = 12                  
        self.mu = 0.6  # Friction coefficient                       
        
        # MPC Weights
        # self.Q = np.diag([150, 150, 150,  200, 200, 300,  1, 1, 1,  1, 1, 1])
        self.Q = np.diag([150, 150, 150,  200, 200, 300,  0.1, 0.1, 0.1,  1, 1, 1])
        # self.Q = np.diag([150, 150, 150,  10, 10, 300,  1, 1, 1,  100, 100, 10])
        self.R = np.eye(self.input_dim) * 1e-10
        
        # Foot and Hip Elements
        self.foot_names = ["FR", "FL", "RR", "RL"]
        self.foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in self.foot_names]
        
        hip_names = ["FR_hip", "FL_hip", "RR_hip", "RL_hip"]
        self.hip_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in hip_names]

        # Gait parameters
        self.q_home = np.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
        self.gait_period = 0.08  # Swing phase duration
        self.step_height = 0.05
        self.last_gait_index = -1
        
        # Real-time interactive parameters
        self.v_des = np.array([0.0, 0.0, 0.0])  # [x_vel, y_vel, yaw_rate]
        self.target_height = 0.28

        # Swing trajectory tracking states
        self.foot_start_pos = [np.zeros(3) for _ in range(4)]
        self.foot_target_pos = [np.zeros(3) for _ in range(4)]
        self.swing_start_time = np.zeros(4)

    def get_state(self):
        quat = self.data.qpos[3:7]
        r = self.quat_to_euler(quat)
        p = self.data.qpos[0:3]
        omega = self.data.qvel[3:6]   
        v = self.data.qvel[0:3]       
        return np.concatenate([r, p, omega, v])

    @staticmethod
    def quat_to_euler(q):
        w, x, y, z = q
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        pitch = np.asin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return np.array([roll, pitch, yaw])

    def get_dynamics_matrices(self, state, foot_positions):
        yaw = state[2]
        cy, sy = np.cos(yaw), np.sin(yaw)
        R_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        
        Ac = np.zeros((12, 12))
        Ac[0:3, 6:9] = R_z.T
        Ac[3:6, 9:12] = np.eye(3)
        
        Bc = np.zeros((12, 12))
        quat = self.data.qpos[3:7]
        R_body = np.zeros((9,))
        mujoco.mju_quat2Mat(R_body, quat)
        R_body = R_body.reshape(3, 3)
        I_world = R_body @ self.I_body @ R_body.T
        I_inv = np.linalg.inv(I_world)
        
        CoM = state[3:6]
        for i in range(4):
            r = foot_positions[i] - CoM
            r_skew = np.array([[0, -r[2], r[1]],
                               [r[2], 0, -r[0]],
                               [-r[1], r[0], 0]])
            Bc[6:9, i*3:(i+1)*3] = I_inv @ r_skew
            Bc[9:12, i*3:(i+1)*3] = np.eye(3) / self.mass
            
        Ad = np.eye(12) + Ac * self.dt_mpc
        Bd = Bc * self.dt_mpc
        
        gd = np.zeros(12)
        gd[11] = -9.81 * self.dt_mpc
        return Ad, Bd, gd

    def get_gait_phase(self, gait_index, leg_idx):
        trot_gait = np.array([
            [GAIT_PHASE_STANCE, GAIT_PHASE_SWING, GAIT_PHASE_SWING, GAIT_PHASE_STANCE],
            [GAIT_PHASE_SWING, GAIT_PHASE_STANCE, GAIT_PHASE_STANCE, GAIT_PHASE_SWING]
        ])
        return trot_gait[gait_index, leg_idx]

    def update_swing_trajectories(self, sim_time, x_curr, foot_positions):
        gait_index = int(sim_time / self.gait_period) % 2
        
        # Project 2D body command velocity to 3D translation in world frame
        yaw = x_curr[2]
        cy, sy = np.cos(yaw), np.sin(yaw)
        v_des_world = np.array([
            cy * self.v_des[0] - sy * self.v_des[1],
            sy * self.v_des[0] + cy * self.v_des[1],
            0.0
        ])
        v_curr_world = x_curr[9:12]

        if gait_index != self.last_gait_index:
            self.last_gait_index = gait_index
            for i in range(4):
                is_stance = (self.get_gait_phase(gait_index, i) == GAIT_PHASE_STANCE)
                if not is_stance:
                    self.foot_start_pos[i] = foot_positions[i].copy()
                    self.swing_start_time[i] = sim_time
                    
                    # Target projected hip position at touchdown
                    hip_pos = self.data.xpos[self.hip_ids[i]].copy()
                    hip_pos_touchdown = hip_pos + self.gait_period * v_curr_world
                    
                    # Raibert Heuristic (Corrected 3D projections)
                    t_stance = self.gait_period
                    self.foot_target_pos[i] = hip_pos_touchdown + (t_stance / 2.0) * v_des_world + 0.03 * (v_curr_world - v_des_world)
                    self.foot_target_pos[i][2] = 0.0 

        q_des = self.data.qpos[7:19].copy()
        qd_des = np.zeros(12)
        
        # Get base motion states to calculate relative velocity
        v_base = self.data.qvel[0:3]
        omega_base = self.data.qvel[3:6]
        CoM = self.data.subtree_com[0]

        for i in range(4):
            is_stance = (self.get_gait_phase(gait_index, i) == GAIT_PHASE_STANCE)
            if is_stance:
                # Keep tracking current joint configurations (avoiding reference jumps)
                q_des[3*i : 3*(i+1)] = self.data.qpos[7 + 3*i : 7 + 3*(i+1)]
                # q_des[3*i : 3*(i+1)] = self.q_home[3*i : 3*(i+1)]
                qd_des[3*i : 3*(i+1)] = 0.0
            else:
                # Swing leg trajectory calculation
                t_swing = sim_time - self.swing_start_time[i]
                phase = np.clip(t_swing / self.gait_period, 0.0, 1.0)
                
                pos_des = (1.0 - phase) * self.foot_start_pos[i] + phase * self.foot_target_pos[i]
                
                # Sine-squared profile for zero touchdown velocity impact
                pos_des[2] = self.step_height * (np.sin(phase * np.pi) ** 2)
                
                # Workspace Jacobian velocity tracking
                jacp = np.zeros((3, self.model.nv))
                jacr = np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.foot_ids[i])
                
                J_leg = jacp[:, 6 + 3*i : 9 + 3*i]
                J_pinv = np.linalg.pinv(J_leg, rcond=1e-4)
                
                error = pos_des - foot_positions[i]
                dq = J_pinv @ error
                q_des[3*i : 3*(i+1)] = self.data.qpos[7 + 3*i : 7 + 3*(i+1)] + 0.6 * dq
                
                # Velocity profiles
                v_des_xy = (self.foot_target_pos[i] - self.foot_start_pos[i]) / self.gait_period
                v_des_z = self.step_height * (np.pi / self.gait_period) * np.sin(2 * phase * np.pi)
                v_des_foot = np.array([v_des_xy[0], v_des_xy[1], v_des_z])
                
                # Relative velocity translation: accounting for base motion
                r_foot = foot_positions[i] - CoM
                v_base_at_foot = v_base + np.cross(omega_base, r_foot)
                
                qd_des[3*i : 3*(i+1)] = J_pinv @ (v_des_foot - v_base_at_foot)
                
        return q_des, qd_des

    def solve_mpc_qp(self, x0, x_ref, foot_positions, gait_schedule):
        """Formulates and solves the stance MPC as a fast QP using qpsolvers."""
        Ad, Bd, gd = self.get_dynamics_matrices(x0, foot_positions)
        
        nx = self.state_dim
        nu = self.input_dim
        N = self.N_horizon
        
        # Dense QP formulation matrices
        A_qp = np.zeros((nx * N, nx))
        B_qp = np.zeros((nx * N, nu * N))
        G_qp = np.zeros(nx * N)
        
        Ad_power = np.eye(nx)
        for i in range(N):
            Ad_power = Ad_power @ Ad
            A_qp[i*nx:(i+1)*nx, :] = Ad_power
            
            if i == 0:
                G_qp[i*nx:(i+1)*nx] = gd
            else:
                G_qp[i*nx:(i+1)*nx] = Ad @ G_qp[(i-1)*nx:i*nx] + gd
                
            for j in range(i + 1):
                if j == i:
                    B_qp[i*nx:(i+1)*nx, j*nu:(j+1)*nu] = Bd
                else:
                    Ad_diff = np.eye(nx)
                    for k in range(i - j):
                        Ad_diff = Ad_diff @ Ad
                    B_qp[i*nx:(i+1)*nx, j*nu:(j+1)*nu] = Ad_diff @ Bd

        Q_stack = np.kron(np.eye(N), self.Q)
        R_stack = np.kron(np.eye(N), self.R)
        X_ref = np.concatenate(x_ref)
        
        e = A_qp @ x0 + G_qp - X_ref
        
        # Objective matrices: 1/2 U^T H U + g^T U
        H = 2 * (B_qp.T @ Q_stack @ B_qp + R_stack)
        H = 0.5 * (H + H.T) + np.eye(nu * N) * 1e-9  # Regularize
        g = 2 * B_qp.T @ Q_stack @ e
        
        # Friction Cone Linear Constraints per leg: |fx| <= mu * fz, |fy| <= mu * fz
        G_leg = np.array([
            [ 1,  0, -self.mu],
            [-1,  0, -self.mu],
            [ 0,  1, -self.mu],
            [ 0, -1, -self.mu]
        ])
        h_leg = np.zeros(4)
        
        G_stack = np.zeros((4 * 4 * N, nu * N))
        h_stack = np.zeros(4 * 4 * N)
        
        lb = np.zeros(nu * N)
        ub = np.zeros(nu * N)
        
        for k in range(N):
            for leg in range(4):
                idx_u = (k * nu) + (leg * 3)
                idx_c = (k * 16) + (leg * 4)
                
                G_stack[idx_c:idx_c+4, idx_u:idx_u+3] = G_leg
                h_stack[idx_c:idx_c+4] = h_leg
                
                is_stance = gait_schedule[k, leg]
                if is_stance == GAIT_PHASE_STANCE:
                    lb[idx_u]   = -50.0  
                    lb[idx_u+1] = -50.0
                    lb[idx_u+2] = 5.0     # Minimum vertical contact force
                    
                    ub[idx_u]   = 50.0
                    ub[idx_u+1] = 50.0
                    ub[idx_u+2] = 150.0   # Maximum vertical contact force
                else:
                    # Swing leg forces must be zero
                    lb[idx_u:idx_u+3] = 0.0
                    ub[idx_u:idx_u+3] = 0.0
                    
        # Fast OSQP solve
        u_opt = qpsolvers.solve_qp(H, g, G_stack, h_stack, lb=lb, ub=ub, solver="osqp")
        
        if u_opt is None:
            return np.zeros(nu)  # Fallback
        return u_opt[:nu]

    def solve_wbc(self, f_mpc, q_des, qd_des, kp, kd):
        nv = self.model.nv
        M = np.zeros((nv, nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)
        h = self.data.qfrc_bias  
        
        J_contacts = []
        for foot_id in self.foot_ids:
            jacp = np.zeros((3, nv))
            jacr = np.zeros((3, nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, foot_id)
            J_contacts.append(jacp)
        J_c = np.vstack(J_contacts)  

        h_j = h[6:18]
        J_cj = J_c[:, 6:18]

        q_curr = self.data.qpos[7:19]
        qd_curr = self.data.qvel[6:18]
        
        tau_feedback = kp * (q_des - q_curr) + kd * (qd_des - qd_curr)
        tau = tau_feedback + h_j - J_cj.T @ f_mpc
        
        return np.clip(tau, -23.7, 23.7)

# --- 2. INTERACTIVE CONTROLLER GUI (MAIN THREAD) ---
def launch_gui(controller, stop_event):
    """Launches the Tkinter GUI on the main thread."""
    root = tk.Tk()
    root.title("Go1 Command Dashboard")
    root.geometry("320x250")
    root.attributes('-topmost', True)

    def on_close():
        stop_event.set()  # Signal simulation thread to exit
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    tk.Label(root, text="Go1 Command Dashboard", font=("Helvetica", 12, "bold")).pack(pady=10)

    tk.Label(root, text="Forward Velocity (X):").pack()
    slider_x = tk.Scale(root, from_=-1.2, to=1.2, resolution=0.1, orient=tk.HORIZONTAL, 
                        command=lambda val: setattr(controller, 'v_des', np.array([float(val), controller.v_des[1], 0.0])))
    slider_x.set(0.0)
    slider_x.pack(fill=tk.X, padx=20)

    tk.Label(root, text="Lateral Velocity (Y):").pack()
    slider_y = tk.Scale(root, from_=-0.5, to=0.5, resolution=0.1, orient=tk.HORIZONTAL, 
                        command=lambda val: setattr(controller, 'v_des', np.array([controller.v_des[0], float(val), 0.0])))
    slider_y.set(0.0)
    slider_y.pack(fill=tk.X, padx=20)

    tk.Label(root, text="Body Target Height:").pack()
    slider_h = tk.Scale(root, from_=0.20, to=0.35, resolution=0.01, orient=tk.HORIZONTAL, 
                        command=lambda val: setattr(controller, 'target_height', float(val)))
    slider_h.set(0.28)
    slider_h.pack(fill=tk.X, padx=20)

    # Monitor simulation status to automatically shut down GUI if viewer window closes
    def check_status():
        if stop_event.is_set():
            root.destroy()
        else:
            root.after(100, check_status)
    
    root.after(100, check_status)
    root.mainloop()

# --- 3. GEOMETRIC VISUALIZATION HELPERS ---
def draw_sphere(viewer, position, size=0.02, rgba=[1, 0, 0, 1]):
    if viewer.user_scn.ngeom >= 1000:
        return
    idx = viewer.user_scn.ngeom
    viewer.user_scn.ngeom += 1
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[idx],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[size, size, size],
        pos=position,
        mat=np.eye(3).flatten(),
        rgba=rgba
    )

# --- 4. PLOTTING FUNCTION ---
def plot_results(history):
    """Plots recorded states, references, and control inputs on shutdown."""
    history = {k: np.array(v) for k, v in history.items()}
    time_vec = history['time']

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # 1. Base Height Tracking
    axs[0].plot(time_vec, history['z_curr'], label='Actual Height', color='b', lw=1.5)
    axs[0].plot(time_vec, history['z_ref'], label='Reference Height', color='r', linestyle='--', lw=1.5)
    axs[0].set_ylabel('Height [m]')
    axs[0].set_title('CoM Tracking Performance')
    axs[0].grid(True)
    axs[0].legend()

    # 2. Base Forward Velocity Tracking
    axs[1].plot(time_vec, history['vx_curr'], label='Actual Vx', color='b', lw=1.5)
    axs[1].plot(time_vec, history['vx_ref'], label='Reference Vx', color='r', linestyle='--', lw=1.5)
    axs[1].set_ylabel('Forward Vel [m/s]')
    axs[1].grid(True)
    axs[1].legend()

    # 3. Vertical MPC Contact Forces
    legs = ["FR", "FL", "RR", "RL"]
    for i in range(4):
        axs[2].plot(time_vec, history['f_mpc'][:, i*3 + 2], label=f'{legs[i]} Fz', alpha=0.8)
    axs[2].set_ylabel('Contact Force [N]')
    axs[2].set_xlabel('Time [s]')
    axs[2].set_title('Vertical Ground Reaction Forces (MPC)')
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.show()

# --- 5. MAIN SIMULATION IN BACKGROUND ---
def run_simulation(controller, stop_event, history_dict):
    # Load model
    xml_path = "/home/aidin/B/mujoco_project/go1_scripts/mujoco_menagerie/unitree_go1/scene.xml" 
    model = controller.model
    data = controller.data
    
    # Initialize standing pose
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    
    trot_gait = np.array([
        [GAIT_PHASE_STANCE, GAIT_PHASE_SWING, GAIT_PHASE_SWING, GAIT_PHASE_STANCE],
        [GAIT_PHASE_SWING, GAIT_PHASE_STANCE, GAIT_PHASE_STANCE, GAIT_PHASE_SWING]
    ])
    
    last_mpc_time = -1.0
    f_mpc = np.zeros(12)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and not stop_event.is_set():
            step_start = time.time()
            sim_time = data.time
            
            # 1. State Acquisition & Local Prediction Frame Assembly
            x_curr = controller.get_state()
            x_ref = []
            
            # Form reference command using correct coordinate heading orientations
            yaw = x_curr[2]
            cy, sy = np.cos(yaw), np.sin(yaw)
            v_des_world = np.array([
                cy * controller.v_des[0] - sy * controller.v_des[1],
                sy * controller.v_des[0] + cy * controller.v_des[1],
                0.0
            ])

            for i in range(controller.N_horizon):
                x_ref.append(np.array([
                    0.0, 0.0, 0.0,              
                    x_curr[3] + v_des_world[0] * (i * controller.dt_mpc), 
                    x_curr[4] + v_des_world[1] * (i * controller.dt_mpc), 
                    controller.target_height,   
                    0.0, 0.0, 0.0,              
                    v_des_world[0], v_des_world[1], 0.0               
                ]))
                
            foot_positions = [data.site_xpos[fid].copy() for fid in controller.foot_ids]
            gait_index = int(sim_time / controller.gait_period) % 2 
            gait_schedule = np.tile(trot_gait[gait_index], (controller.N_horizon, 1))

            # 2. 66 Hz OSQP MPC Loop
            if sim_time - last_mpc_time >= controller.dt_mpc or last_mpc_time < 0:
                f_mpc = controller.solve_mpc_qp(x_curr, x_ref, foot_positions, gait_schedule)
                last_mpc_time = sim_time

            # 3. Kinematic Swings
            q_des, qd_des = controller.update_swing_trajectories(sim_time, x_curr, foot_positions)
            
            # Gain scheduling
            kp = np.zeros(12)
            kd = np.zeros(12)
            for i in range(4):
                is_stance = (controller.get_gait_phase(gait_index, i) == GAIT_PHASE_STANCE)
                if is_stance:
                    # Stance gains (very soft tracking, letting MPC force dictate stance behavior)
                    kp[3*i : 3*(i+1)] = 5.0
                    kd[3*i : 3*(i+1)] = 0.2
                else:
                    # Swing tracking gains
                    kp[3*i : 3*(i+1)] = 150.0
                    kd[3*i : 3*(i+1)] = 10.0

            # 4. Torque Execution
            torques = controller.solve_wbc(f_mpc, q_des, qd_des, kp, kd)
            data.ctrl[:] = torques
            
            # 5. Physics Step
            mujoco.mj_step(model, data)

            # Record tracking states for plotting (at 100 Hz rate)
            if int(sim_time * 1000) % 10 == 0:
                history_dict['time'].append(sim_time)
                history_dict['z_curr'].append(x_curr[5])
                history_dict['z_ref'].append(controller.target_height)
                history_dict['vx_curr'].append(x_curr[9])
                history_dict['vx_ref'].append(v_des_world[0])
                history_dict['f_mpc'].append(f_mpc.copy())

            # 6. Debug Rendering
            viewer.user_scn.ngeom = 0  
            for ref_state in x_ref:
                draw_sphere(viewer, ref_state[3:6], size=0.015, rgba=[0, 1, 0, 0.4])
            for target in controller.foot_target_pos:
                draw_sphere(viewer, target, size=0.018, rgba=[0, 0.5, 1, 0.7])
            for pos in foot_positions:
                draw_sphere(viewer, pos, size=0.012, rgba=[1, 0, 0, 0.8])

            viewer.sync()
            
            # Sync timing to step
            time_to_sleep = model.opt.timestep - (time.time() - step_start)
            # if time_to_sleep > 0:
            #     time.sleep(time_to_sleep)
                
        stop_event.set()  # Guarantee all threads stop on exit

# --- 6. PROGRAM START ---
if __name__ == "__main__":
    xml_path = "/home/aidin/B/mujoco_project/go1_scripts/mujoco_menagerie/unitree_go1/scene.xml" 
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    controller = Go1Controller(model, data)
    stop_event = threading.Event()
    
    # Store records for plot
    history = {
        'time': [],
        'z_curr': [], 'z_ref': [],
        'vx_curr': [], 'vx_ref': [],
        'f_mpc': []
    }
    
    # Start physics simulation in background thread
    sim_thread = threading.Thread(
        target=run_simulation, 
        args=(controller, stop_event, history), 
        daemon=True
    )
    sim_thread.start()
    
    # Run GUI on Main Thread (safe for OS graphics handlers)
    launch_gui(controller, stop_event)
    
    # Wait for simulation to safely shut down
    sim_thread.join()
    print("Simulation stopped. Loading diagnostic plots.")
    
    # Display the results
    plot_results(history)
