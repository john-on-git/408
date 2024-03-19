from time import sleep
import pygame
from environments import MazeEnv
from agents import *
import tensorflow as tf
from keras import layers

env = MazeEnv(render_mode="human", nCoins=10, gameLength=100)
agent = PPOAgent(
    actionSpace=env.ACTION_SPACE,
    hiddenLayers=[
        layers.Reshape((6,6,1)),
        layers.Conv2D(1,2,strides=(1,1), activation=tf.nn.sigmoid),
        layers.Conv2D(1,2,strides=(1,1), activation=tf.nn.sigmoid),
        layers.Dense(8, activation=tf.nn.sigmoid),
        layers.Dense(16, activation=tf.nn.sigmoid),
        layers.Flatten()
    ],
    validActions=env.validActions,
    learningRate=0
)
agent.load_weights("checkpoints\MazeEnv_PPOAgent.tf")

observation = tf.expand_dims(tf.convert_to_tensor(env.reset()[0]),0)
TICK_RATE_HZ = 10
tickDelay = 1/TICK_RATE_HZ
countDownLength = 1 * TICK_RATE_HZ
endCountDown = countDownLength
announcedEnding = False

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
            print(f"Agent starved. Score: {env.score}")
            observation = tf.expand_dims(tf.convert_to_tensor(env.reset()[0]),0)
            announcedEnding = True
        endCountDown-=1
    elif env.terminated:
        if not announcedEnding:
            print(f"Time up! Score: {env.score}")
            observation = tf.expand_dims(tf.convert_to_tensor(env.reset()[0]),0)
            announcedEnding = True
        endCountDown-=1
    else:
        observation, reward, _, _, _ = env.step(agent.act(observation))
        observation = tf.expand_dims(tf.convert_to_tensor(observation),0)