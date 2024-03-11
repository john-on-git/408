from time import sleep
import pygame
from environments import MazeEnv

env = MazeEnv(render_mode="human", nCoins=10, gameLength=25)

while not env.terminated and not env.truncated:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        elif event.type == pygame.KEYDOWN:
            match(event.key):
                case pygame.K_ESCAPE:
                    exit()
                case pygame.K_UP:
                    env.step(0)
                case pygame.K_LEFT:
                    env.step(1)
                case pygame.K_DOWN:
                    env.step(2)
                case pygame.K_RIGHT:
                    env.step(3)
    sleep(.1)
print("Score:", env.score)