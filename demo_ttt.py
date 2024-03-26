from time import sleep
from environments import TTTEnv, Team
from agents import *
import pygame
import tensorflow as tf

def opponentStart():
    action = opponent.act(tf.expand_dims(tf.convert_to_tensor(env.calcLogits(Team.CROSS)),0))
    env.board[int(action/env.size)][action%env.size] = Team.CROSS
    env.notify()

env = TTTEnv(render_mode="human", opponent=None)
opponent = DQNAgent(epsilon=.1, epsilonDecay=1,learningRate=0, hiddenLayers=[layers.Flatten(), layers.Dense(8,activation=tf.nn.sigmoid)], discountRate=0, actionSpace=env.actionSpace, validActions=env.validActions)
opponent.load_weights("checkpoints/demo_for_submission/TTTEnv_DQNAgent_seed0.tf")
env.opponent = opponent

env.reset()
opponentStart() #CPU goes first
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
                    _, reward, _, _, _ , = env.step(action) #take action
                else:
                    print("Invalid move.")
    sleep(.1)
    if env.terminated or env.truncated:
        sleep(1)
        _ = env.reset()
        opponentStart() #CPU goes first