from time import sleep
from environments import TTTEnv, Team, TTTSearchAgent
from agents import *
import pygame
import tensorflow as tf

def opponentAct():
    action = opponent.act(tf.expand_dims(tf.convert_to_tensor(env.calcLogits(Team.CROSS)),0))
    if env.board[int(action/env.size)][action%env.size] == Team.EMPTY: #enact move if valid
        s, _, terminated, truncated, _ = env.halfStep(Team.CROSS, action) #handle CPU action
    else:
        raise Exception("Tic-Tac-Toe opponent made invalid move.")
    return (s, None, terminated, truncated, {})

env = TTTEnv(render_mode="human", opponent=None)
opponent = AdvantageActorCriticAgent(epsilon=0, learningRate=0, hiddenLayers=[layers.Flatten(), layers.Dense(16,activation=tf.nn.relu)], discountRate=0, actionSpace=env.actionSpace, validActions=env.validActions)
opponent.load_weights("checkpoints/TTTEnv_A2C_DEMO.tf")
env.opponent = opponent

env.reset()
opponentAct() #CPU goes first
while True:
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
                    env.step(action) #take action
                else:
                    print("Invalid move.")
    sleep(.1)
    if env.terminated or env.truncated:
        sleep(1)
        _ = env.reset()
        opponentAct() #CPU goes first