from time import sleep
from environments import TagEnv
from agents import *
import tensorflow as tf
from keras import layers
import pygame

env = TagEnv(render_mode="human")
agent = REINFORCEAgent(
    learningRate=0,
    actionSpace=env.actionSpace,
    hiddenLayers=[layers.Dense(4, activation=tf.nn.sigmoid)],
    validActions=env.validActions,
    epsilon=0,
    epsilonDecay=0,
    discountRate=0
)
#agent = AdvantageActorCriticAgent(learningRate=0, actionSpace=env.ACTION_SPACE, hiddenLayers=[layers.Flatten(), layers.Dense(16, activation=tf.nn.sigmoid),layers.Dense(32, activation=tf.nn.sigmoid)])
agent.load_weights("checkpoints\TagEnv_REINORCEAgent.tf")
TICK_RATE_HZ = 100
tickDelay = 1/TICK_RATE_HZ
countDownLength = 1
endCountDown = countDownLength
announcedEnding = False

observation = tf.expand_dims(tf.convert_to_tensor(env.reset()[0]),0)
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
    
    sleep(tickDelay)

    #win and loss logic
    if endCountDown == 0:
        env.reset()
        announcedEnding = False
        endCountDown = countDownLength
    elif env.truncated:
        if not announcedEnding:
            announcedEnding = True
            print(f"Agent crashed.")
        endCountDown-=1
    elif env.terminated:
        if not announcedEnding:
            announcedEnding = True
            print("Agent won.")
        endCountDown-=1
    else:
        observation = tf.expand_dims(tf.convert_to_tensor(env.step(agent.act(observation))[0]),0)
