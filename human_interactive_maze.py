from time import sleep
import pygame
from environments import MazeEnv

env = MazeEnv(render_mode="human", nCoins=0)
rewardThisEpisode = 0
totalReward = 0
nEpisodes = 0
running = True
while running:
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
    if running:
        totalReward+=rewardThisEpisode
        print("reward this episode:", rewardThisEpisode)
        nEpisodes+=1
        rewardThisEpisode = 0
        env.reset()
print("num episodes:", nEpisodes)
print("average reward:", 0 if nEpisodes==0 else totalReward/nEpisodes)