#template from https://www.pygame.org/docs/
import pygame
from threading import Thread
from jgw_cs408.observer import Observable, Observer
import math

class Entity():
    def __init__(self, rect:pygame.Rect, rotation:float=0) -> None:
        self.rect = rect
        self.rotation = rotation
    def getCenter(self):
        return self.rect.center
class Vehicle(Entity):
    def __init__(self, rect:pygame.Rect, rotation:float, speed=5, rotationSpeed=math.pi/120) -> None:
        super().__init__(rect, rotation)
        self.speed = speed
        self.rotationSpeed = rotationSpeed #in radians
    def turn(self, amount) -> None:
        self.rotation = (self.rotation+amount) % (2*math.pi) #rotation repeats, in radians
    def left(self):
        self.turn(-self.rotationSpeed)
    def right(self):
        self.turn(self.rotationSpeed)
    def advance(self):
        self.rect.center+=pygame.math.Vector2([math.cos(self.rotation)*self.speed, math.sin(self.rotation)*self.speed])
class RaceModel(Observable):
    def __init__(self) -> None:
        super().__init__()
        #load hitbox masks
        coneHitboxFactory       = pygame.image.load("jgw_cs408/img/cone.png").get_rect
        carHitboxFactory        = pygame.image.load("jgw_cs408/img/car.png").get_rect
        finishHitboxFactory     = pygame.image.load("jgw_cs408/img/finish.png").get_rect
        checkpointHitboxFactory = pygame.image.load("jgw_cs408/img/checkpoint.png").get_rect

        #parameters to reset to when RaceModel.reset() is called
        self.playerVehicleInitialPosition = (0,0)
        self.playerVehicleInitialRotation = math.radians(270)

        self.playerVehicle = Vehicle(rect=carHitboxFactory(), rotation=self.playerVehicleInitialRotation)
        self.playerVehicle.rect.center = self.playerVehicleInitialPosition = (0,0)
        #TODO
            #declare obstacle factory
            #read track data
            #for position in track data
                #newObstacle = obstacleFactory.factory()
        self.obstacles = []
        self.checkpoints = []
        self.finishes = []
        self.vehicles = [self.playerVehicle]
        self.terminated = False
        self.truncated = False

        #define the course
        def coneCircle(model:RaceModel, center:pygame.Vector2, radius:float, gapBetweenCones:int=50):
            circumference = math.pi * radius * 2
            numberOfCones = math.ceil(circumference/gapBetweenCones/2)
            radiansBetweenCones = (2*math.pi)/numberOfCones
            for i in range(numberOfCones):
                offset = pygame.math.Vector2([math.cos(i*radiansBetweenCones)*radius, math.sin(i*radiansBetweenCones)*radius])
                obstacle = Entity(rect=coneHitboxFactory())
                obstacle.rect.center = (center.x+offset.x, center.y+offset.y)
                model.obstacles.append(obstacle)

        #coneCircle(self, pygame.Vector2(1175,0), 1000.0)
        #coneCircle(self, pygame.Vector2(1175,0), 1350.0)
        coneCircle(self, pygame.Vector2(425,0), 250.0)
        coneCircle(self, pygame.Vector2(425,0), 600.0)

        for i in range(0,300,100):
            obstacle = Entity(rect=coneHitboxFactory())
            obstacle.rect.center = (-100+i, 100)
            self.obstacles.append(obstacle)
        
        for i in range(0,300,50):
            finish = Entity(rect=finishHitboxFactory())
            finish.rect.center = (-100+i, 155)
            self.finishes.append(finish)
    def reset(self) -> tuple[tuple, int, bool, bool, None]:
        self.playerVehicle.rect.center = self.playerVehicleInitialPosition = (0,0)
        self.playerVehicle.rotation = self.playerVehicleInitialRotation
        self.terminated = False
        self.truncated = False
        return (self.calcLogits(), 0, self.terminated, self.truncated, None)
    def step(self, action : (0|1|2) = 1) -> tuple[tuple, int, bool, bool, None]:
        reward = 1 #baseline per frame
        if not self.terminated and not self.truncated:
            #update the vehicle's angle according to the action
            match action:
                case 0: #turn left
                    self.playerVehicle.left()
                case 1: #no action
                    pass
                case 2: #turn right
                    self.playerVehicle.right()
                case _:
                    raise ValueError() if type(action) is int else TypeError()
            #update the vehicle's position
            self.playerVehicle.advance()
            #check for collisions
            if self.playerVehicle.rect.collidelist([x.rect for x in self.obstacles]) != -1: #with obstacles
                self.truncated = True
            if self.playerVehicle.rect.collidelist([x.rect for x in self.finishes]) != -1: #with finish line
                self.terminated = True
                reward+=100
        logits = self.calcLogits()
        info = None
        return (logits, reward, self.terminated, self.truncated, info)
    def calcLogits(self) -> list[float]:
        pass
    def notify(self) -> None:
        observer: Observer
        for observer in super().getObservers():
            observer.update(self)
#view
class RaceView(Observer):
    def __init__(self, model : RaceModel, resolution) -> None:
        super().__init__()
        self.BG_COLOR = pygame.color.Color(126,126,126)
        pygame.init()
        displayResolution = resolution
        self.screen = pygame.display.set_mode(displayResolution)
        
        #TODO
        self.scale = 1

        #load images
        coneSurface   = pygame.transform.scale_by(pygame.image.load("jgw_cs408/img/cone.png"), self.scale)
        finishSurface = pygame.transform.scale_by(pygame.image.load("jgw_cs408/img/finish.png"), self.scale)
        carSurface    = pygame.transform.scale_by(pygame.image.load("jgw_cs408/img/car.png"), self.scale)

        #init drawing data structures
        self.cameraPos = model.playerVehicle.getCenter
        self.centerPos = pygame.Vector2(resolution[0]/2, resolution[1]/2)

        self.obstacles   = []
        self.finishes    = []
        self.vehicles    = []
        self.drawTargets = [(finishSurface, self.finishes),(coneSurface, self.obstacles),(carSurface, self.vehicles)] #order controls the layering
        
        self.clock = pygame.time.Clock()
        self.running = False
    def main(self):
        while self.running:
            # fill the screen with a color to wipe away anything from last frame
            self.screen.fill(self.BG_COLOR)
            #draw everything
            for surface, entities in self.drawTargets:
                for entity in entities:
                    rotatedSurface = pygame.transform.rotate(surface, -math.degrees(entity.rotation))
                    surfaceRect = rotatedSurface.get_rect()
                    centering = pygame.math.Vector2(surfaceRect.width/2, surfaceRect.height/2)
                    screenPosition = ((entity.getCenter()-centering-self.cameraPos())) + self.centerPos
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
        if type(world) is RaceModel:
            self.obstacles.clear()
            self.finishes.clear()
            self.vehicles.clear()
            for obstacle in world.obstacles:
                self.obstacles.append(obstacle)
            for finish in world.finishes:
                self.finishes.append(finish)
            for vehicle in world.vehicles:
                self.vehicles.append(vehicle)

class RaceEnv():
    def reset(self, seed:int=None) -> None:
        return self.model.reset()
    def __init__(self, render_mode : (None|str)=None) -> None:
        self.model = RaceModel()
        self.actionSpace = [0,1,2]
        if (render_mode=="human"):
            self.view = RaceView(resolution=pygame.Vector2(750,500), model=self.model)
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