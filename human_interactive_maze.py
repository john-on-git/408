from time import sleep
import pygame
from environments import MazeEnv
import keyboard

env = MazeEnv(render_mode="human", nCoins=10)

keyboard.on_press_key('w', lambda _: (env.step(0)))
keyboard.on_press_key('a', lambda _: (env.step(1)))
keyboard.on_press_key('s', lambda _: (env.step(2)))
keyboard.on_press_key('d', lambda _: (env.step(3)))
keyboard.on_press_key('esc', lambda _: exit())

while not env.terminated and not env.truncated:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
    sleep(.1)
print("Score:", env.score)