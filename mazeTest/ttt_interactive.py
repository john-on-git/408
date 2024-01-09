from time import sleep
from ttt_env import TTTEnv
import keyboard

env = TTTEnv(render_mode="human")
keyboard.on_press_key('7', lambda _: (env.step(0)))
keyboard.on_press_key('8', lambda _: (env.step(1)))
keyboard.on_press_key('9', lambda _: (env.step(2)))
keyboard.on_press_key('4', lambda _: (env.step(3)))
keyboard.on_press_key('5', lambda _: (env.step(4)))
keyboard.on_press_key('6', lambda _: (env.step(5)))
keyboard.on_press_key('1', lambda _: (env.step(6)))
keyboard.on_press_key('2', lambda _: (env.step(7)))
keyboard.on_press_key('3', lambda _: (env.step(8)))
keyboard.on_press_key('esc', lambda _: exit())

while True:
    sleep(0.1)
    if env.model.terminated or env.model.truncated:
        exit()