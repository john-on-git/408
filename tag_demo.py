from time import sleep
from environments import TagEnv
from agents import AdvantageActorCriticAgent, REINFORCE_MENTAgent
import tensorflow as tf
from keras import layers
import pygame

env = TagEnv(render_mode="human")
agent = AdvantageActorCriticAgent(
    actionSpace=env.ACTION_SPACE,
    hiddenLayers=[layers.Flatten(), layers.Dense(16, activation=tf.nn.sigmoid),layers.Dense(32, activation=tf.nn.sigmoid)],
    validActions=env.validActions,
    learningRate=0
)

#agent = AdvantageActorCriticAgent(learningRate=0, actionSpace=env.ACTION_SPACE, hiddenLayers=[layers.Flatten(), layers.Dense(16, activation=tf.nn.sigmoid),layers.Dense(32, activation=tf.nn.sigmoid)])
agent.load_weights("checkpoints\TagEnv_AdvantageActorCriticAgent.tf")
TICK_RATE_HZ = 100
tickDelay = 1/TICK_RATE_HZ
countDownLength = 1 * TICK_RATE_HZ
endCountDown = countDownLength
announcedEnding = False

observation = tf.expand_dims(tf.convert_to_tensor(env.reset()[0]),0)
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()

    observation = tf.expand_dims(tf.convert_to_tensor(env.step(agent.act(observation))[0]),0)
    
    sleep(tickDelay)

    #win and loss logic
    if env.truncated:
        if not announcedEnding:
            announcedEnding = True
            print("Agent crashed.")
        endCountDown-=1
    elif env.terminated:
        if not announcedEnding:
            announcedEnding = True
            print("Agent won.")
        endCountDown-=1

    
    if endCountDown == 0:
        env.reset()
        announcedEnding = False
        endCountDown = countDownLength
