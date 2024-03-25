import pygame
from environments import MazeEnv
import time
from agents import OptimalMazeAgent

rngSeed = 0
env = MazeEnv()
agent = OptimalMazeAgent()

terminated = False
truncated = False
rewardOverall = 0
rewardThisEpisode = 0
running = True
currentEpisode=0
terminated = False
truncated = False
N_EPISODES = 100
while running and currentEpisode<N_EPISODES:
    rewardThisEpisode = 0
    observation = env.reset()[0]
    rngSeed+=1
    while running and not (terminated or truncated):
        observation, reward, terminated, truncated, _ = env.step(agent.act(observation))
        rewardThisEpisode+=reward
        observation = observation
        time.sleep(0.1)
    print(f"reward (episode {currentEpisode+1}):", rewardThisEpisode)
    currentEpisode+=1
    rewardOverall+=rewardThisEpisode
    terminated = False
    truncated = False
    observation, _ = env.reset(rngSeed)
print("num episodes:", N_EPISODES)
print("average reward:", rewardOverall/N_EPISODES)
