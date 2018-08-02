# Change the path to load gym lib
import sys
sys.path.append('./gym')
import gym
env = gym.make('MountainCarContinuous-v0')

for i_episode in range(20):
  observation = env.reset()
  for t in range(50):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
      print("Episode finished after {} timesteps".format(t+1))
      break
