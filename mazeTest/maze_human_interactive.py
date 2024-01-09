from time import sleep
from maze_env import MazeEnv
import keyboard

env = MazeEnv(render_mode="human")
percept = None
running = True
keyboard.on_press_key('w', lambda _: (env.step(0)))
keyboard.on_press_key('a', lambda _: (env.step(1)))
keyboard.on_press_key('s', lambda _: (env.step(2)))
keyboard.on_press_key('d', lambda _: (env.step(3)))
keyboard.on_press_key('esc', lambda _: exit())

while running:
    sleep(0.1)