"""Renders the PandaPickAndPlace-v3 environment in a window and steps it with random actions for 200 steps so the robot arm's motion can be watched live."""

import gymnasium as gym
import panda_gym
import time

#gymnasism is the framework for an agent making an environment - interface that simulators use
#panda_gym is the environment that a robot arm uses in trainng

env = gym.make("PandaPickAndPlace-v3", render_mode="human")

#building the actual world and rendermode allows me to view it in a window

obs, info = env.reset()

for step in range(200):
    action = env.action_space.sample()
    #random move part
    obs, reward, terminated, truncated, info = env.step(action)
    #actually moves the arm (obs reward etc are the outputs of the action in line 15)
    time.sleep(0.05)
    #makes the motions slower so I can see them
    if terminated or truncated:
        obs, info = env.reset()

env.close()
#closes it after 200 steps are done