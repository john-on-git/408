from time import sleep
from ttt_env import TTTEnv, SearchAgent, Team
#from jgw_cs408.agents import *
import keyboard
import pygame

env = TTTEnv(render_mode="human", opponent=None, size=5)
opponent = SearchAgent(depth=1)#DQNAgent(epsilon=0, learningRate=0, discountRate=0, actionSpace=env.actionSpace)
#opponent.load_weights("jgw_cs408/checkpoints/TTTParallelDQNAgent.tf")
env.opponent = opponent

#keyboard.on_press_key('7', lambda _: (env.step(0)))
#keyboard.on_press_key('8', lambda _: (env.step(1)))
#keyboard.on_press_key('9', lambda _: (env.step(2)))
#keyboard.on_press_key('4', lambda _: (env.step(3)))
#keyboard.on_press_key('5', lambda _: (env.step(4)))
#keyboard.on_press_key('6', lambda _: (env.step(5)))
#keyboard.on_press_key('1', lambda _: (env.step(6)))
#keyboard.on_press_key('2', lambda _: (env.step(7)))
#keyboard.on_press_key('3', lambda _: (env.step(8)))
keyboard.on_press_key('esc', lambda _: exit())

_ = env.opponentAct(opponent) #CPU goes first
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if pygame.mouse.get_pressed()[0]: #if it was a left click
                x,y = pygame.mouse.get_pos()
                #screen coords to grid coords
                x=int(x/env.view.xSize)
                y=int(y/env.view.ySize)
                #grid coord to action
                action = x + y*env.model.size
                if env.model.board[y][x] == Team.EMPTY:
                    env.step(action) #take action
                else:
                    print("Invalid move.")
    sleep(.1)
    if env.model.terminated or env.model.truncated:
        sleep(3)
        _ = env.reset()
        _ = env.opponentAct(opponent) #CPU goes first