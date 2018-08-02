# Change the path to load gym lib
import sys
sys.path.append('./gym')
import gym
import numpy as np
import random
import math
from random import Random
env = gym.make('CartPole-v0')


THETA_DOT_THRESHOLD = 12 * 2 * math.pi / 360
THETA_THRESHOLD = 2.4
THETA_DOT_STATE_SPACE_SIZE = 5
THETA_STATE_SPACE_SIZE = 3
LEARNING_RATE = 0.5
GAMMA = 0.9
EPSILON = 0.5

rand = Random()

v_table = [[ 2*[0.0] for _ in range(THETA_DOT_STATE_SPACE_SIZE) ] for _ in range(THETA_STATE_SPACE_SIZE)]

def updateVtable(newState, lastState, action, reward):
  newState_q_list = []
  for act in [0, 1]:
    newState_q_list.append(v_table[newState[0]][newState[1]][act])
  v_table[lastState[0]][lastState[1]][action] += LEARNING_RATE*(reward+GAMMA*max(newState_q_list)-v_table[lastState[0]][lastState[1]][action])
  return None

def getState(observation):
  theta_state = max(min(observation[2], THETA_THRESHOLD), -THETA_THRESHOLD)
  theta_state = np.floor((theta_state+THETA_THRESHOLD)/(2*THETA_THRESHOLD/THETA_STATE_SPACE_SIZE)).astype(int)
  theta_dot_state = max(min(observation[2], THETA_DOT_THRESHOLD), -THETA_DOT_THRESHOLD)
  theta_dot_state = np.floor((theta_dot_state+THETA_DOT_THRESHOLD)/(2*THETA_DOT_THRESHOLD/THETA_DOT_STATE_SPACE_SIZE)).astype(int)
  return (max(min(theta_state, THETA_STATE_SPACE_SIZE-1), 0), max(min(theta_dot_state, THETA_DOT_STATE_SPACE_SIZE-1), 0))

def getAction(state):
  if (rand.uniform(0.0,1.0) < EPSILON):
    action = rand.randint(0,1)
  else:
    if v_table[state[0]][state[1]][0] > v_table[state[0]][state[1]][1]: 
      action =  0
    else: action = 1
  return action

for i_episode in range(1000):
  observation = env.reset()
  end_episode = False
  if i_episode > 500:
    EPSILON = 0
  for t in range(200):
    env.render()
    old_state = getState(observation)
    action = getAction(old_state)
    # action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    new_state = getState(observation)
    updateVtable(new_state, old_state, action, reward)
    if done and not end_episode:
      end_episode = True
      print(v_table)
      print("Episode {}".format(i_episode))
      print("Episode finished after {} timesteps".format(t+1))
      break
