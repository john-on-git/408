#template from https://www.pygame.org/docs/
import pygame
from threading import Thread
from jgw_cs408.observer import Observable, Observer
import math
import random

class Entity():
    def __init__(self, rect:pygame.Rect, rotation:float=0) -> None:
        self.rect = rect
        self.rotation = rotation
    def getCenter(self):
        return self.rect.center
class Mover(Entity):
    def __init__(self, rect:pygame.Rect, rotation:float, speed=5, rotationRate=math.pi/120) -> None:
        super().__init__(rect, rotation)
        self.speed = speed
        self.rotationRate = rotationRate #in radians
    def turn(self, amount) -> None:
        self.rotation = (self.rotation+amount) % (2*math.pi) #rotation repeats, in radians
    def left(self):
        self.turn(-self.rotationRate)
    def right(self):
        self.turn(self.rotationRate)
    def advance(self):
        x = math.cos(self.rotation)*self.speed
        y = math.sin(self.rotation)*self.speed
        self.rect = self.rect.move(x,y)
class TagModel(Observable):
    def __init__(self, nSeekers, speedRatio, maxTime, arenaX, arenaY) -> None:
        super().__init__()
        self.scale = 10
        runnerSpeed = 5
        runnerRotationRate = math.pi/30
        #load hitbox masks
        runnerHitboxFactory         = pygame.transform.scale_by(pygame.image.load("jgw_cs408/img/runner.png"), self.scale).get_rect
        self.seekerHitboxFactory    = pygame.transform.scale_by(pygame.image.load("jgw_cs408/img/seeker.png"), self.scale).get_rect

        #parameters to reset to when RaceModel.reset() is called
        self.runnerInitialPosition = (arenaX/2 * self.scale, arenaY/2 * self.scale)
        self.runnerInitialRotation = math.radians(90)
        self.nSeekers = nSeekers
        self.seekerSpeed = runnerSpeed * speedRatio

        #init entities
        self.arena = Entity(pygame.Rect(0,0,arenaX*self.scale,arenaY*self.scale), 0) #game ends if agent is not in contact with this rect
        self.runner = Mover(rect=runnerHitboxFactory(), rotation=self.runnerInitialRotation, speed=runnerSpeed * self.scale, rotationRate=runnerRotationRate)
        self.runner.rect.center = self.runnerInitialPosition
        self.runners = [self.runner]
        self.seekers = []
        for _ in range(self.nSeekers):
            seeker = Mover(rect=self.seekerHitboxFactory(), rotation=0, speed=self.seekerSpeed * self.scale)
            seeker.rect.center = self.genSeekerPosition() #TODO random positioning
            self.seekers.append(seeker)

        self.terminated = False
        self.truncated = False
        self.time = 0
        self.maxTime = maxTime
    def genSeekerPosition(self,dist=None):
        dist = random.randint(100*self.scale,200*self.scale) if dist==None else dist
        x,y = self.runner.rect.center
        angle = random.random() * 360
        x += math.cos(angle)*dist
        y += math.sin(angle)*dist
        return (x,y)
    def reset(self) -> tuple[list[float], int, bool, bool, None]:
        self.runner.rect.center = self.runnerInitialPosition
        self.runner.rotation = self.runnerInitialRotation

        #reset seekers
        self.seekers.clear()
        for _ in range(self.nSeekers):
            seeker = Mover(rect=self.seekerHitboxFactory(), rotation=0, speed=self.seekerSpeed * self.scale)
            seeker.rect.center = self.genSeekerPosition() #TODO random positioning
            self.seekers.append(seeker)

        self.terminated = False
        self.truncated = False
        self.time = 0
        self.notify() #redraw
        return (self.calcLogits(), None)
    def step(self, action : (0|1|2) = 1) -> tuple[tuple, int, bool, bool, None]:
        reward = 1 #baseline per step
        if not self.terminated and not self.truncated:
            self.time+=1
            #update the runner's angle according to the action
            match action:
                case 0: #turn left
                    self.runner.left()
                case 1: #no action
                    pass
                case 2: #turn right
                    self.runner.right()
                case _:
                    raise ValueError() if type(action) is int else TypeError()
            #update the runner's position
            self.runner.advance()
            #update the seekers' angles & positions
            for seeker in self.seekers:
                dx = self.runner.rect.center[0] - seeker.rect.center[0]
                dy = self.runner.rect.center[1] - seeker.rect.center[1]
                seeker.rotation = math.atan2(dy, dx)
                seeker.advance()
            #check for collisions
                
            if self.runner.rect.collidelist([seeker.rect for seeker in self.seekers]) != -1:
                self.truncated = True
            elif not self.runner.rect.colliderect(self.arena): #with obstacles
                self.truncated = True
            elif self.time>=self.maxTime: #and time out
                self.terminated = True
        logits = self.calcLogits()
        info = None
        self.notify() #redraw
        return (logits, reward, self.terminated, self.truncated, info)
    def calcLogits(self) -> list[float]:
        x = [float(self.runner.rect.center[0]), float(self.runner.rect.center[1]), self.runner.rotation]
        for seeker in self.seekers:
            x.extend([float(seeker.rect.center[0]), float(seeker.rect.center[1])])
        return x
    def notify(self) -> None:
        observer: Observer
        for observer in super().getObservers():
            observer.update(self)
#view
class TagView(Observer):
    def __init__(self, model : TagModel, resolution) -> None:
        super().__init__()
        self.BG_COLOR = pygame.color.Color(126,126,126)
        pygame.init()
        displayResolution = resolution
        self.screen = pygame.display.set_mode(displayResolution)
        
        #TODO
        self.scale = 1/model.scale

        #load images
        runnerSurface = pygame.image.load("jgw_cs408/img/runner.png")
        seekerSurface = pygame.image.load("jgw_cs408/img/seeker.png")
        arenaSurface  = pygame.Surface(pygame.Vector2(model.arena.rect.size) * self.scale)
        arenaSurface.fill((255,255,255))

        #init drawing data structures
        self.cameraPos = model.runner.getCenter
        self.centerPos = pygame.Vector2(resolution[0]/2, resolution[1]/2)

        self.runners = []
        self.seekers = []
        self.arenas = [model.arena]
        self.drawTargets = [(arenaSurface, self.arenas),(seekerSurface, self.seekers), (runnerSurface, self.runners)] #order controls the layering
        
        self.clock = pygame.time.Clock()
        self.running = False
    def main(self):
        while self.running:
            # fill the screen with a color to wipe away anything from last frame
            self.screen.fill(self.BG_COLOR)
            #draw everything
            for surface, entities in self.drawTargets:
                for entity in entities:
                    center = pygame.math.Vector2(entity.getCenter()) * self.scale #screen space position of object
                    cameraPos = pygame.math.Vector2(self.cameraPos()) * self.scale #adjust relative to camera

                    rotatedSurface = pygame.transform.rotate(surface, -math.degrees(entity.rotation))
                    surfaceRect = rotatedSurface.get_rect()
                    centering = pygame.math.Vector2(surfaceRect.width/2, surfaceRect.height/2)
                    
                    screenPosition = self.centerPos + center - cameraPos - centering
                    self.screen.blit(rotatedSurface, screenPosition)
            
            # flip() the display to put your work on screen
            pygame.display.flip()
        exit()
    def open(self):
        self.running = True
        self.thread = Thread(target=self.main, daemon=True)
        self.thread.start()
    def close(self):
        self.running = False
        self.thread.join()
        pygame.quit()
    def update(self, world):
        if type(world) is TagModel:
            for _, xs in self.drawTargets:
                xs.clear()
            for runner in world.runners:
                self.runners.append(runner)
            for seeker in world.seekers:
                self.seekers.append(seeker)
            self.arenas.append(world.arena)

class TagEnv():
    def reset(self, seed:int=None) -> None:
        return self.model.reset()
    def __init__(self, render_mode : (None|str)=None, nSeekers = 1, speedRatio = 2/3, maxTime=200, arenaX=500, arenaY=500) -> None:
        self.model = TagModel(nSeekers, speedRatio, maxTime, arenaX, arenaY)
        self.actionSpace = [0,1,2]
        if (render_mode=="human"):
            self.view = TagView(resolution=pygame.Vector2(750,500), model=self.model)
            self.model.addObserver(self.view)
            self.model.notify()
            self.view.open()
        else:
            self.view = None
    def step(self, action):
        return self.model.step(action)
    def close(self):
        if not self.view == None:
            self.view.close()
            self.view = None