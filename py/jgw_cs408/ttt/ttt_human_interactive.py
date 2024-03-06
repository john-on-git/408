from time import sleep
from ttt_env import TTTEnv, TTTSearchAgent, Team
from jgw_cs408.agents import *
import keyboard
import pygame
import tensorflow as tf

def opponentAct():
    action = env.model.opponent.act(tf.expand_dims(tf.convert_to_tensor(env.model.calcLogits(Team.CROSS)),0))
    if env.model.board[int(action/env.model.size)][action%env.model.size] == Team.EMPTY: #enact move if valid
        s, _, terminated, truncated, _ = env.model.halfStep(Team.CROSS, action) #handle CPU action
    else:
        raise Exception("Tic-Tac-Toe opponent made invalid move.")
    return (s, None, terminated, truncated, {})

env = TTTEnv(render_mode="human", opponent=None, size=3)
opponent = TTTSearchAgent(epsilon=.75) #DQNAgent(epsilon=0, learningRate=0, discountRate=0, actionSpace=env.actionSpace)
#opponent.load_weights("jgw_cs408/checkpoints/TTTParallelDQNAgent.tf")
env.model.opponent = opponent

keyboard.on_press_key('esc', lambda _: exit())

#_ = opponentAct() #CPU goes first
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        elif event.type == pygame.MOUSEBUTTONDOWN: #this must be in the main thread due to pygame shenanigans
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
        sleep(1)
        _ = env.reset()
        #opponentAct() #CPU goes first