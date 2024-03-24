from time import sleep
from environments import TagEnv
import pygame


env = TagEnv(render_mode="human", nSeekers=0, arenaDimensions=(1000,1000))

#redraw rate
TICK_RATE_HZ = 100
ACTION_DELAY = 3
tickDelay = 1/TICK_RATE_HZ

#input handling 
action = 1
timeout = None

#restart delay
endCountDownLength = 1 * TICK_RATE_HZ
endCountDown = None

#metrics
rewardOverall = 0
rewardThisEpoch = 0
N_EPISODES = 20

for i in range(N_EPISODES):
    while endCountDown!=0:
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

        _, reward, _, _, _ = env.step(action)
        rewardThisEpoch+=reward

        #handle action time out
        if timeout==0:
            action = 1
            timeout = None
        elif timeout != None:
            timeout-=1

        #handle winning/losing
        if env.truncated and endCountDown == None:
            endCountDown = endCountDownLength
            print("You crashed!")
        elif env.terminated and endCountDown == None:
            endCountDown = endCountDownLength
            print("You won!")
        elif endCountDown != None:
            endCountDown-=1
        sleep(tickDelay)
    env.reset()
    endCountDown = None
    rewardOverall+=rewardThisEpoch
    rewardThisEpoch=0
print("num episodes:", N_EPISODES)
print("average reward:", rewardOverall/N_EPISODES)