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

{video}

