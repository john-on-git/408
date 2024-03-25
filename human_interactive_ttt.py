from time import sleep
from environments import TTTEnv, Team, TTTSearchAgent
from agents import *
import pygame
import tensorflow as tf

env = TTTEnv(render_mode="human")

env.reset()
terminated = False
truncated = False
rewardOverall = 0
rewardThisEpisode = 0
running = True
currentEpisode=0
terminated = False
truncated = False
N_EPISODES = 25

while running and currentEpisode<N_EPISODES:
    rewardThisEpisode = 0
    while running and not (terminated or truncated):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            elif event.type == pygame.KEYDOWN:
                match(event.key):
                    case pygame.K_ESCAPE:
                        exit()
            elif event.type == pygame.MOUSEBUTTONDOWN: #this must be in the main thread due to pygame shenanigans
                if pygame.mouse.get_pressed()[0]: #if it was a left click
                    x,y = pygame.mouse.get_pos()
                    #screen coords to grid coords
                    x=int(x/env.view.xSize)
                    y=int(y/env.view.ySize)
                    #grid coord to action
                    action = x + y*env.size
                    if env.board[y][x] == Team.EMPTY:
                        _, reward, terminated, truncated, _ = env.step(action) #take action
                        rewardThisEpisode+=reward
                    else:
                        print("Invalid move.")
        sleep(.1)
    if truncated and reward>=(10**env.size):
        print("You Won!")
    elif truncated:
        print("You Lost. ):")
    else:
        print("Draw.")
    print(f"reward (episode {currentEpisode+1}):", rewardThisEpisode)
    currentEpisode+=1
    rewardOverall+=rewardThisEpisode
    sleep(1)
    terminated = False
    truncated = False
    _ = env.reset()
env.view.close()
print("num episodes:", N_EPISODES)
print("average reward:", rewardOverall/N_EPISODES)
