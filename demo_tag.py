from time import sleep
from environments import TagEnv
from agents import *
import tensorflow as tf
from keras import layers
import pygame

TICK_RATE_HZ = 100
tickDelay = 1/TICK_RATE_HZ

env = TagEnv(render_mode="human")

#change me!
agent = PPOAgent(
    learningRate=0,
    actionSpace=env.actionSpace,
    hiddenLayers=[layers.Dense(4, activation=tf.nn.sigmoid)],
    validActions=env.validActions,
    epsilon=0,
    epsilonDecay=0,
    discountRate=0
)
agent.load_weights("demo_checkpoints/TagEnv_PPOAgent_seed0.tf")

while True:
    terminated = False
    truncated = False
    observation = tf.expand_dims(tf.convert_to_tensor(env.reset()[0]),0)
    rewardThisEpoch = 0
    while not (terminated or truncated):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
        sleep(tickDelay)
        observation, reward, terminated, truncated, _ = env.step(agent.act(observation))
        rewardThisEpoch+=reward
        observation = tf.expand_dims(tf.convert_to_tensor(observation),0)
    print("Reward this epoch:",rewardThisEpoch)