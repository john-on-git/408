from time import sleep
import pygame
from environments import MazeEnv
from agents import *
import tensorflow as tf
from keras import layers

env = MazeEnv(render_mode="human")

#change me!
agent = PPOAgent(
    actionSpace=env.actionSpace,
    hiddenLayers=[layers.Conv2D(4,3,(1,1)), layers.MaxPool2D((2,2)), layers.Flatten(), layers.Dense(16), layers.Dense(16), layers.Dense(16), layers.Dense(16), layers.Dense(16), layers.Dense(16)],
    validActions=env.validActions,
    learningRate=.0,
    epsilon=0.2,
    epsilonDecay=0,
    discountRate=0
)
agent.load_weights("demo_checkpoints\MazeEnv_PPOAgent_10k.tf")

observation = tf.expand_dims(tf.convert_to_tensor(env.reset()[0]),0)
TICK_RATE_HZ = 10
tickDelay = 1/TICK_RATE_HZ
countDownLength = 1 * TICK_RATE_HZ
endCountDown = countDownLength
announcedEnding = False
rewardThisEpisode = 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
    
    sleep(tickDelay)

    #win and loss logic
    if endCountDown == 0:
        rewardThisEpisode = 0
        env.reset()
        announcedEnding = False
        endCountDown = countDownLength
    elif env.truncated:
        if not announcedEnding:
            print(f"Agent starved. Score: {rewardThisEpisode}")
            observation = tf.expand_dims(tf.convert_to_tensor(env.reset()[0]),0)
            announcedEnding = True
        endCountDown-=1
    elif env.terminated:
        if not announcedEnding:
            print(f"Time up! Score: {rewardThisEpisode}")
            observation = tf.expand_dims(tf.convert_to_tensor(env.reset()[0]),0)
            announcedEnding = True
        endCountDown-=1
    else:
        observation, reward, _, _, _ = env.step(agent.act(observation))
        rewardThisEpisode+=reward
        observation = tf.expand_dims(tf.convert_to_tensor(observation),0)