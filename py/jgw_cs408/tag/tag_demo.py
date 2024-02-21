from time import sleep
from tag_env import TagEnv
from jgw_cs408.agents import *
import pygame

env = TagEnv(render_mode="human")
agent = RandomAgent(env.actionSpace)
TICK_RATE_HZ = 100
tickDelay = 1/TICK_RATE_HZ
countDownLength = 2 * TICK_RATE_HZ
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
    if env.model.truncated:
        if not announcedEnding:
            announcedEnding = True
            print("Agent crashed!")
        endCountDown-=1
    elif env.model.terminated:
        if not announcedEnding:
            announcedEnding = True
            print("Agent won!")
        endCountDown-=1

    
    if endCountDown == 0:
        env.reset()
        announcedEnding = False
        endCountDown = countDownLength