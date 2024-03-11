from time import sleep
from environments import TagEnv
import pygame


env = TagEnv(render_mode="human", maxTime=-1, arenaDimensions=(1000,1000))
action = 1
timeout = None
TICK_RATE_HZ = 100
ACTION_DELAY = 3
tickDelay = 1/TICK_RATE_HZ
countDownLength = 2 * TICK_RATE_HZ
endCountDown = countDownLength
announcedEnding = False

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        elif event.type == pygame.KEYDOWN:
            match(event.key):
                case pygame.K_ESCAPE:
                    exit()
                case pygame.K_LEFT:
                    action = 0
                case pygame.K_RIGHT:
                    action = 2
        elif event.type == pygame.KEYUP:
            match(event.key):
                case pygame.K_LEFT:
                    timeout = ACTION_DELAY
                case pygame.K_RIGHT:
                    timeout = ACTION_DELAY

    env.step(action)
    if timeout==0:
        action = 1
        timeout = None
    elif timeout != None:
        timeout-=1
    sleep(tickDelay)

    #win and loss logic
    if env.truncated:
        if not announcedEnding:
            announcedEnding = True
            print("You crashed!")
        endCountDown-=1
    elif env.terminated:
        if not announcedEnding:
            announcedEnding = True
            print("You won!")
        endCountDown-=1

    
    if endCountDown == 0:
        env.reset()
        announcedEnding = False
        endCountDown = countDownLength