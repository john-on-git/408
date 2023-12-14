from time import sleep
from maze_env import MazeEnv, Action
import keyboard

env = MazeEnv(renderMode="human")
percept = None
running = True
keyboard.on_press_key('w', lambda _: env.step(Action.UP))
keyboard.on_press_key('a', lambda _: env.step(Action.LEFT))
keyboard.on_press_key('s', lambda _: env.step(Action.DOWN))
keyboard.on_press_key('d', lambda _: env.step(Action.RIGHT))
keyboard.on_press_key('esc', lambda _: exit()) #huh?

while running:
    sleep(0.1)