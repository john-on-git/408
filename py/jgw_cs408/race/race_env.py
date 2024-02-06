#template from https://www.pygame.org/docs/
import pygame
from enum import Enum
from threading import Thread
from jgw_cs408.observer import Observable, Observer
import math

#TODO
    #collision from image (requires fixing transparency)
    #how to define track? Cones?

class Entity():
    def __init__(self, rect, position:pygame.math.Vector2) -> None:
        self.rect = rect
        self.position = position
class Vehicle(Entity):
    def __init__(self, rect, rotation, speed=1, rotationSpeed=0.314) -> None:
        super.__init__(rect)
        self.speed = speed
        self.rotation = rotation
        self.rotationSpeed = rotationSpeed #in radians
    def turn(self, amount) -> None:
        self.rotation+=amount
        self.rotation%=2*math.pi #rotation repeats, in radians
    def left(self):
        self.turn(-self.rotationSpeed)
    def right(self):
        self.turn(self.rotationSpeed)
    def advance(self):
        self.position+=pygame.math.Vector2([math.acos(self.rotation)*self.speed, math.asin(self.rotation)*self.speed])
class RaceModel(Observable):
    def __init__(self) -> None:
        super().__init__()
        self.vehicle = Vehicle()
        self.track = None #TODO
    def reset(self) -> None:
        pass
    def step(self, action : (0|1|2) = 1) -> (tuple, int, bool, bool, None):
        #update the vehicle's angle according to the action
        match action:
            case 0: #turn left
                self.vehicle.left()
            case 1: #no action
                pass
            case 2: #turn right
                self.vehicle.right()
            case _:
                raise ValueError() if type(action) is int else TypeError()
        #update the vehicle's position
        self.vehicle.advance()
        #check for collisions
        reward = None
        logits = self.calcLogits()
        terminated = None
        truncated = None
        info = None
        return (logits, reward, terminated, truncated, info)
    def calcLogits(self) -> [float]:
        pass
    def notify(self) -> None:
        observer: Observer
        for observer in super().getObservers():
            observer.update(self)
#view
class RaceView(Observer):
    def __init__(self, model : RaceModel, resolution) -> None:
        super().__init__()
        self.BG_COLOR = pygame.color.Color(255,255,255)
        pygame.init()
        displayResolution = resolution
        self.screen = pygame.display.set_mode(displayResolution)
        
        #TODO
        self.xScale = None
        self.yScale = None

        #load images
        coneImg = pygame.transform.scale(pygame.image.load("jgw_cs408/img/cone.png"), (self.xScale, self.yScale))
        finishImg = pygame.transform.scale(pygame.image.load("jgw_cs408/img/greenPerson.png"), (self.xScale, self.yScale))
        carImg = pygame.transform.scale(pygame.image.load("jgw_cs408/img/greenPerson.png"), (self.xScale, self.yScale))
        #init surfaces
        coneSurface = pygame.Surface()
        finishSurface = pygame.Surface()
        carSurface = pygame.Surface()
        #init rects
        #init entities

        car = Vehicle(rect=carRect, position=pygame.math.Vector2(0,0), rotation=0)
        self.cameraPos = self.car.position
        cones = [] #TODO define track
        
        self.drawTargets = [(coneSurface, cones),(finishSurface, [finishLine]),(carSurface, [car])] #order controls the layer order

        self.clock = pygame.time.Clock()
        self.running = False
    def main(self):
        while self.running:
            # fill the screen with a color to wipe away anything from last frame
            self.screen.fill(self.BG_COLOR)

            #draw everything
            for sprite, entities in self.drawTargets: 
                for entity in entities:
                    self.screen.blit(sprite, entity.position+self.cameraPos)
            
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
        pass
class RaceEnv():
    def reset(self,seed:int=None) -> None:
        self.model.reset()
        return (self.model.calcLogits(), None)
    def __init__(self, nCoins:int=3, startPosition:(str|tuple)="random", render_mode : (None|str)=None) -> None:
        self.nCoins=nCoins
        self.startPosition = startPosition
        self.model = RaceModel()
        if (render_mode=="human"):
            self.view = RaceView(resolution=(500,500), model=self.model)
            self.model.addObserver(self.view)
            self.view.open()
        else:
            self.view = None
    def step(self, action):
        if not self.view == None:
            # pygame.QUIT event means the user clicked X to close your window
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
        return self.model.step(action)
    def close(self):
        if not self.view == None:
            self.view.close()
            self.view = None