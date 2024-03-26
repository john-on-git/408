from time import sleep
import pygame
from environments import MazeEnv
from agents import *
import tensorflow as tf
from keras import layers

env = MazeEnv(render_mode="human")
agent = ActorCriticAgent(
    actionSpace=env.actionSpace,
    hiddenLayers=[layers.Conv2D(4,3,(1,1)), layers.Flatten()],
    validActions=env.validActions,
    learningRate=.0,
    epsilon=0,
    epsilonDecay=0,
    discountRate=0
)
agent.load_weights("checkpoints\demo_for_submission\MazeEnv_ActorCriticAgent_seed3000.tf")

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