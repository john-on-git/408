from time import sleep
from jgw_cs408.environments.tag_env import TagEnv
import keyboard
import pygame

def setAction(input):
    global nextAction
    global timeout
    match input:
        case 'a':
            nextAction = 0
        case 'd':
            nextAction = 2
        case _:
            nextAction = 1
    timeout = 10

env = TagEnv(render_mode="human")
env.model.maxTime=9999999
nextAction = 1
timeout = 10
TICK_RATE_HZ = 100
tickDelay = 1/TICK_RATE_HZ
countDownLength = 2 * TICK_RATE_HZ
endCountDown = countDownLength
announcedEnding = False
keyboard.on_press_key('a', lambda _: setAction('a'))
keyboard.on_press_key('d', lambda _: setAction('d'))

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
    env.step(nextAction)
    if timeout==0:
        nextAction = 1
    else:
        timeout-=1
    sleep(tickDelay)

    #win and loss logic
    if env.model.truncated:
        if not announcedEnding:
            announcedEnding = True
            print("You crashed!")
        endCountDown-=1
    elif env.model.terminated:
        if not announcedEnding:
            announcedEnding = True
            print("You won!")
        endCountDown-=1

    
    if endCountDown == 0:
        env.reset()
        announcedEnding = False
        endCountDown = countDownLength
    
    #if endCountDown==0:
    #    exit()