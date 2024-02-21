from time import sleep
from ttt_env import TTTEnv, SearchAgent, Team
#from jgw_cs408.agents import *
import keyboard
import pygame

env = TTTEnv(render_mode="human", opponent=None, size=3)
opponent = SearchAgent()#DQNAgent(epsilon=0, learningRate=0, discountRate=0, actionSpace=env.actionSpace)
#opponent.load_weights("jgw_cs408/checkpoints/TTTParallelDQNAgent.tf")
env.opponent = opponent

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