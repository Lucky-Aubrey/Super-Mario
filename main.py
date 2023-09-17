# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 01:23:29 2023

@author: Lucky
"""

# Setup Mario
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
import gym

# # Create environment
# env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")

# env = JoypadSpace(env, SIMPLE_MOVEMENT)

# done = True
# env.reset()
# for step in range(5000):
#     if done:
#        env.reset()
       
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     done = terminated or truncated

#     env.render()

# env.close()

# Preprocessing
from gym.wrappers import GrayScaleObservation
# Import vectorization wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# Matplotlib
from matplotlib import pyplot as plt

# 1. Create base environment
env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
# 2. Simplify controls
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# 3. Grayscale
env = GrayScaleObservation(env, keep_dim=True)
# 4. wrap inside dummy environmnet
env = DummyVecEnv([lambda: env])
# 5. stack the frames
env = VecFrameStack(env, 4, channels_order='last')

# Traing the RL model
# import os for file path management
import os
# import PPO for algorithms
from stable_baselines3 import PPO
# import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback

class TrainAndLoggingCallback(BaseCallback):
    
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        
    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
            
        return True

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

# Model saving
callback = TrainAndLoggingCallback(1000, save_path=CHECKPOINT_DIR)

# Setup Model
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.0001, 
            n_steps=512)

# Train model
model.learn(total_timesteps=10000, callback=callback)

# Load model
model = PPO.load('./train/best_model_10000')
# Start game
state = env.reset()
# Loop through the game
while True:
    
    action, _state = model.predict(state, deterministic=False)
    obs, reward, terminated, info = env.step(action)
    
    env.render()
    