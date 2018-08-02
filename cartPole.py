# Change the path to load gym lib
import sys
sys.path.append('./gym')
import gym
import numpy as np
import random
import math
from random import Random
env = gym.make('CartPole-v0')
rand = Random()

X_MAX, X_DOT_MAX, THETA_MAX, THETA_DOT_MAX = env.observation_space.high
X_MIN, X_DOT_MIN, THETA_MIN, THETA_DOT_MIN = env.observation_space.low

X_SPACE_SIZE = 3
X_DOT_SPACE_SIZE = 3
THETA_SPACE_SIZE = 5
THETA_DOT_SPACE_SIZE = 3

LEARNING_RATE = 0.5
GAMMA = 0.9
EPSILON = 0.6
TRAIN_EPISODES = 300
MAX_TIMESTEPS = 200

v_table = [[[[ 2*[0.0] 
            for _ in range(THETA_DOT_SPACE_SIZE) ] 
            for _ in range(THETA_SPACE_SIZE) ]
            for _ in range(X_DOT_SPACE_SIZE) ]
            for _ in range(X_SPACE_SIZE) ]

def getByTuple(l, t):
  return l[t[0]][t[1]][t[2]][t[3]][t[4]]

def setByTuple(l, t, n):
  l[t[0]][t[1]][t[2]][t[3]][t[4]] = n

def updateVtable(newState, lastState, action, reward):
  newState_q_list = []
  for act in [0, 1]: newState_q_list.append(getByTuple(v_table, newState + (act,)))
  new_value = (1-LEARNING_RATE)*getByTuple(v_table, lastState+(action,)) + LEARNING_RATE*(reward+GAMMA*max(newState_q_list))
  setByTuple(v_table, lastState + (action,), new_value)
  return None

def getState(observation):
  # Get state of x
  x_state = max(min(observation[0], X_MAX), X_MIN)
  x_state = np.floor((x_state+X_MAX)/(2*X_MAX/X_SPACE_SIZE)).astype(int)
  if x_state >= X_SPACE_SIZE: x_state = X_SPACE_SIZE - 1
  # if observation[0] < 0: x_state = 0
  # else: x_state = 1

  # Get state of x dot
  if observation[1] < -0.5: x_dot_state = 0
  elif observation[1] < 0.5: x_dot_state = 1
  else: x_dot_state = 2

  # Get state of theta
  theta_state = max(min(observation[2], THETA_MAX), THETA_MIN)
  theta_state = np.floor((theta_state+THETA_MAX)/(2*THETA_MAX/THETA_SPACE_SIZE)).astype(int)
  if theta_state >= THETA_SPACE_SIZE: theta_state = THETA_SPACE_SIZE - 1
  # if observation[2] < 0: theta_state = 0
  # else: theta_state = 1

  # Get state of theta dot
  if observation[3] < -0.5: theta_dot_state = 0
  elif observation[3] < 0.5: theta_dot_state = 1
  else: theta_dot_state = 2

  return (x_state, x_dot_state, theta_state, theta_dot_state)

def getAction(state):
  if (rand.uniform(0.0, 1.0) < EPSILON):
    action = rand.randint(0, 1)
  else:
    if getByTuple(v_table, state + (0,)) > getByTuple(v_table, state + (1,)): 
      action = 0
    else: action = 1
  return action

def train(max_episodes, max_timestep):
  for i_episode in range(max_episodes):
    observation = env.reset()
    end_episode = False
    for t in range(max_timestep):
      env.render()
      old_state = getState(observation)
      action = getAction(old_state)
      observation, reward, done, info = env.step(action)
      new_state = getState(observation)
      updateVtable(new_state, old_state, action, reward)
      # print(v_table)
      if done and not end_episode:
        end_episode = True
        print("Episode {}".format(i_episode))
        print("Episode finished after {} timesteps".format(t+1))
        break

def greedy(state):
  if getByTuple(v_table, state + (0,)) > getByTuple(v_table, state + (1,)): 
    return 0
  else:
    return 1

def play():
  observation = env.reset()
  timestep = 0
  while True:
    timestep += 1
    env.render()
    state = getState(observation)
    action = greedy(state)
    observation, reward, done, info = env.step(action)
    if done:
      print("Failed after {} timesteps".format(timestep))
      timestep = 0
      observation = env.reset()

print("Start training at %d training episodes and %d max timesteps" % (TRAIN_EPISODES, MAX_TIMESTEPS))
train(TRAIN_EPISODES, MAX_TIMESTEPS)
print("Training is completed, start executing the policy.")
play()

