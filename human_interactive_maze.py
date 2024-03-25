from time import sleep
import pygame
from environments import MazeEnv

env = MazeEnv(startPosition=[(0,0)],render_mode="human")
rewardOverall = 0
N_EPISODES = 25
running = True
currentEpisode=0
while running and currentEpisode<N_EPISODES:
    rewardThisEpisode = 0
    while running and not env.terminated and not env.truncated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                match(event.key):
                    case pygame.K_ESCAPE:
                        running = False
                    case pygame.K_UP:
                        _, reward, _, _, _ = env.step(0)    
                        rewardThisEpisode += reward
                    case pygame.K_LEFT:
                        _, reward, _, _, _ = env.step(1)
                        rewardThisEpisode += reward
                    case pygame.K_DOWN:
                        _, reward, _, _, _ = env.step(2)
                        rewardThisEpisode += reward
                    case pygame.K_RIGHT:
                        _, reward, _, _, _ = env.step(3)
                        rewardThisEpisode += reward
    print(f"reward (episode {currentEpisode+1}):", rewardThisEpisode)
    currentEpisode+=1
    rewardOverall+=rewardThisEpisode
    env.reset()
env.view.close()
print("num episodes:", N_EPISODES)
print("average reward:", 0 if N_EPISODES==0 else rewardOverall/N_EPISODES)