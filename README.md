# go1-locomotion-playground
I'm experimenting with go1 and try to learn and implement as I go forward and test different training methods.

here I try to train the unitree go1 robot to walk (or gait) in mujoco environment.
I have downloaded the go1 xml file from the mujoco menagerie of Google DeepMind. they already provide a scene.xml so to have the environment ready. the only thing I modified is that I added four sensors for the contact points of go1 feets, I assume it's important for the training policy to have a knowledge about when any of the feets are in contact with the ground otherwise it would feel like moving in the air.

the go1 joints are already set to position PD control so there is no need to modify them from torque to position control in XML file.

to train the robot, I use PyTorch and StableBaseline3 but the problem is the training loop will be on cpu since the Mujoco is. here the recomendation would be to use JAX and it would be awsome but I tried it apart from the fact that it required some specific function declerations it wasn't that fast on my PC. Not saying it's the JAX issue but my implementations and internal problems or the fact that I'm runing from WSL2 ubuntu and not native but anyway since I have Nvidia GPU available I switched to Warp. 

Why am I going through all this instead of just using google playground python library? well it's broken.. even the Colab Tutorial does not work due to some Error Raise from Warp-Lang where packages libraries are incompatible at the moment (or I'm novice) either way since I'm not a Python Developer I try to solve this for my case and after some tweaking the Mujoco-warp and warp are installed and I can do the computations of mujoco steps in miliseconds instead of minutes using capture method.

Now trying a simple RL implementation results in a very poor trained model as the Go1 tries to move forward by any means and it looks miserable. 

https://github.com/user-attachments/assets/3d34984e-3129-4f5d-bf98-709db60e1612

but to improve the model I tried a couple of times after failing everytime I decided to check whether there are any available trained models so I can understand what is the issue with my reward function 
so after examining the related github repos I came across one where I adapted it for my code and trained for 20M timesteps and it looks like this
this time training is for random velocity and orientation commands so that we can control it in simulation:



https://github.com/user-attachments/assets/2eceb027-0c00-480c-a1cd-b98b71b83f70

Could not hardcode gait for the feet to move in certain phase, intersting to see the policy pick up a simple efficient trotting along training the robot although is feels like the simulation is in fast forward.
A little bit tweaking the frequencies and timesteps result in slightly slower simulations yet the feet move slightly making it like the Go1 is sliding.

https://github.com/user-attachments/assets/8c4b9915-cd60-44be-bd51-143812ed878d



### Phase 1: The Initial Prototype (Fixed Forward Walking)

The initial phase focused on verifying that the GPU-parallelized physics loop worked correctly. It proved that 2,048 environments could run simultaneously on a single GPU without crashing or memory leaks.

* **Task**: Walk forward at a constant target velocity ($v_{x\_target} = 0.5 \text{ m/s}$, $v_y = 0$, $\omega_{yaw} = 0$).
* **State Representation (37 Dimensions)**: Includes local linear/angular velocities, gravity vector projection, joint position errors, joint velocities, and direct touch contact feedback.
* **Touch Sensor Handling**: Read foot forces from the `sensordata` buffer and normalized them using an exponential function:
  $$\text{contact} = 1.0 - \exp(-\frac{\text{force}}{50.0})$$
* **Control Space**: Coarse PD targets applied with a relatively high action scale of `0.4`.
* **Training Dynamics**: Evaluated using a large policy step horizon (`n_steps=128`), producing huge mini-batches of $262,144$ transitions per policy epoch. This stabilized gradient updates but resulted in slower training iterations.

---

### Phase 2: Current Implementation (Dynamic Command Tracking)

The current phase upgrades the setup to support omnidirectional walking using a dynamic command-tracking paradigm inspired by standard quadruped learning frameworks.

* **Task**: Robust tracking of randomized command vectors $[v_x, v_y, \omega_{yaw}]$ generated dynamically and resampled individually for each environment every $6$ seconds (300 environment steps).
* **State Representation (48 Dimensions)**: Replaced foot touch sensors with the active commands $[v_x, v_y, \omega_{yaw}]$ and added the previous step's actions to the observation tensor to help the policy smooth out target transitions.
* **Tuned Control Bounds**: Lowered the action scale from `0.4` to `0.25`. This narrower control window helps prevent the joints from shaking or moving too rapidly, protecting the virtual motors.
* **Comprehensive Legged-Gym Reward Structure**: Replaced basic rewards with a more complete set of locomotion metrics, including:
  * **Joint Acceleration Penalty**: Minimizes abrupt velocity changes by tracking velocities over time.
  * **Estimated Torque Penalty**: A surrogate metric derived from PD error:
    $$\tau \approx K_p(q^* - q) - K_d \dot{q}$$
  * **Action Smoothness Penalty**: Discourages rapid output fluctuations between consecutive control steps.
* **Optimized PPO Training Schedule**: Decreased the policy step horizon (`n_steps=32`) to run faster policy updates, matching empirical locomotion training setups.

---

## 🛠️ Key GPU Optimizations

This repository uses several key optimization techniques to maximize training throughput:

1. **Zero-Copy Memory Mapping**: Using `wp.to_torch`, the raw Warp CUDA memory addresses of the simulator's states (`qpos`, `qvel`, `sensordata`) are mapped directly to PyTorch tensors. This completely eliminates CPU-GPU data transfer overhead during training loops.
2. **CUDA Graph Capture**: Steps through the simulator $10$ times (`frame_skip = 10` for a $50\text{ Hz}$ control loop) to warm up and compile the underlying GPU kernels. These are captured in a static CUDA graph, bypassing runtime execution delays.
3. **In-place Vectorized Resets**: Instead of resetting the entire simulation, only individual environments that hit termination states (such as falling over or experiencing numerical errors) are reset. Their configurations are re-written directly on the GPU without pausing the other active environments.








https://github.com/user-attachments/assets/70c4f661-5da0-4f8b-9669-ded29c901ed6


