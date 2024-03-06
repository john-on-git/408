#template from https://www.pygame.org/docs/
import pygame
from enum import Enum
from threading import Thread
from jgw_cs408.observer import Observable, Observer
from random import Random

SCORE_PER_COIN = 1
FOOD_PER_COIN = 10
INITIAL_FOOD = FOOD_PER_COIN * 2

#models
class Square(Enum):
    EMPTY = 0,
    SOLID = 1

class Entity():
    def __init__(self, coords) -> None:
        self.coords = coords
class Coin(Entity):
    def __init__(self, coords) -> None:
        super().__init__(coords)

class MazeModel(Observable):
    def __init__(self, squares : list[list[Square]], startPosition:(str|tuple), nCoins:int, gameLength:int=None, seed:int=None) -> None:
        super().__init__()
        #assert shape is ok (all rows must be the same length)
        WIDTH = len(squares[0])
        for row in squares:
            assert len(row) == WIDTH
        #init constants
        self.SQUARES = squares
        self.INITIAL_PLAYER_POSITION = startPosition
        self.GAME_LENGTH = gameLength
        self.N_COINS = nCoins
        self.EMPTY_SQUARES = [] #pre-calculated for placing entities
        for y in range(len(self.SQUARES)):
            for x in range(len(self.SQUARES[y])):
                if self.SQUARES[y][x] == Square.EMPTY:
                    self.EMPTY_SQUARES.append((y,x))
        self.reset(seed)
    def reset(self, seed:int=None) -> None:
        self.random = Random(seed)
        self.score = 0
        self.food = INITIAL_FOOD
        self.time = 0
        self.terminated = False
        self.truncated = False
        self.coins = []
        if self.INITIAL_PLAYER_POSITION=="random":
            self.placePlayer()
        else:
            self.PLAYER_AVATAR = Entity(self.INITIAL_PLAYER_POSITION)
        for _ in range(self.N_COINS): #add coins
            self.placeCoin()
        self.notify() #update view
        return (self.calcLogits(), None)
    def step(self, action:(0|1|2|3|4)) -> tuple[list[float], int, bool, bool, dict]:
        def canMoveTo(coords : tuple) -> bool:
            y,x = coords
            return x>=0 and y>=0 and y<len(self.SQUARES) and x<len(self.SQUARES[0]) and self.SQUARES[y][x] == Square.EMPTY
        reward = 0
        if not self.terminated and not self.truncated:
            self.food-=1
            self.time+=1 #advance time
            markedForDelete = [] #things to be deleted this step
            #move player avatar
            match action:
                case 0:
                    target = (self.PLAYER_AVATAR.coords[0]-1, self.PLAYER_AVATAR.coords[1]) #up
                case 1:
                    target = (self.PLAYER_AVATAR.coords[0], self.PLAYER_AVATAR.coords[1]-1) #left
                case 2:
                    target = (self.PLAYER_AVATAR.coords[0]+1, self.PLAYER_AVATAR.coords[1]) #down
                case 3:
                    target = (self.PLAYER_AVATAR.coords[0], self.PLAYER_AVATAR.coords[1]+1) #right
                case 4:
                    target = self.PLAYER_AVATAR.coords                                      #pass
                case _:
                    raise Exception("Got invalid action "+str(action)+". Expected (0|1|2|3|4).")

            if canMoveTo(target):
                self.PLAYER_AVATAR.coords = target
            #coin collection, spawn new coin
                for coin in self.coins:
                    if coin.coords == self.PLAYER_AVATAR.coords:
                        markedForDelete.append(coin)
                        reward+=SCORE_PER_COIN
                        self.score+=SCORE_PER_COIN
            for entity in markedForDelete:
                if type(entity) == Coin:
                    self.coins.remove(entity)
                    self.placeCoin() #remember to replace it
                    self.food+=FOOD_PER_COIN
            #check for loss
            if self.GAME_LENGTH is not None and self.time>=self.GAME_LENGTH:
                self.terminated = True #end of game because of time out
            elif self.food<=0:
                self.truncated = True #end of game because the player died
        #reward = reward
        logits = self.calcLogits()
        info = {}
        self.notify() #update view
        return (logits, reward, self.terminated, self.truncated, info)
    def calcLogits(self) -> list[float]:
        LOGIT_EMPTY  = 0.0
        LOGIT_SOLID  = 1.0
        LOGIT_PLAYER = 2.0
        LOGIT_COIN   = 3.0
        def calcLogit(y: int,x : int) -> float:
            if self.SQUARES[y][x] == Square.SOLID:
                return LOGIT_SOLID
            elif self.PLAYER_AVATAR.coords == (y,x):
                return LOGIT_PLAYER
            else:
                for coin in self.coins:
                    if coin.coords == (y,x):
                        return LOGIT_COIN
                return LOGIT_EMPTY
        #construct logits from world
        logits = []
        for y in range(len(self.SQUARES)):
            logits.append([None] * len(self.SQUARES))
            for x in range(len(logits[0])):
                logits[y][x] = calcLogit(y,x)
        return logits
    def placePlayer(self) -> tuple:
        emptySquares = self.EMPTY_SQUARES.copy()
        for xs in [self.coins]:
            for x in xs:
                emptySquares.remove(x.coords)
        self.PLAYER_AVATAR = Entity(coords=self.random.choice(emptySquares))
    def placeCoin(self) -> None:
        validForCoin = self.EMPTY_SQUARES.copy() #list of empty squares
        validForCoin.remove(self.PLAYER_AVATAR.coords) #can't place on player
        for xs in [self.coins]: #or on top of another coin
            for x in xs:
                validForCoin.remove(x.coords)
        self.coins.append(Coin(coords=self.random.choice(validForCoin)))
    def notify(self) -> None:
        observer: Observer
        for observer in super().getObservers():
            observer.update(self)
#view
class MazeView(Observer):
    def __init__(self, world : MazeModel, resolution) -> None:
        super().__init__()
        pygame.init()
        displayResolution = resolution
        self.screen = pygame.display.set_mode(displayResolution)
        self.xSize = displayResolution[0]/len(world.SQUARES[0]) #horizontal size of squares
        self.ySize = displayResolution[1]/len(world.SQUARES) #vertical size of squares
        self.squares = [None, []]
        self.players = [None, []]
        self.coins   = [None, []]
        self.enemies = [None, []]

        #square surface
        self.squares[0] = pygame.Surface((self.xSize+1,self.ySize+1))
        self.squares[0].fill(pygame.color.Color(255,255,255))
        
        #player surface
        self.players[0] = pygame.transform.scale(pygame.image.load("jgw_cs408/img/greenPerson.png"), (self.xSize, self.ySize)) #TODO transparency doesn't work, probably a file format thing

        #coin surface
        self.coins[0] = pygame.transform.scale(pygame.image.load("jgw_cs408/img/coin.png"), (self.xSize, self.ySize)) #TODO transparency doesn't work, probably a file format thing
        
        #create squares
        for y in range(len(world.SQUARES)):
            for x in range(len(world.SQUARES[y])):
                if(world.SQUARES[y][x] == Square.EMPTY):
                    self.squares[1].append((x*self.xSize, y*self.ySize))

        #get player location
        self.players[1].clear()
        self.players[1].append((world.PLAYER_AVATAR.coords[1]*self.xSize, world.PLAYER_AVATAR.coords[0]*self.ySize))
        
        #get coin locations
        self.coins[1].clear()
        for viewModel, entities in [(self.coins, world.coins)]:
            for entity in entities:
                y,x = entity.coords
                viewModel[1].append((x*self.xSize, y*self.ySize))

        self.clock = pygame.time.Clock()
        self.running = False
    def main(self):
        while self.running:
            # fill the screen with a color to wipe away anything from last frame
            self.screen.fill(pygame.color.Color(70,70,70))

            #draw surfaces
            #for (thing, desc) in [(self.squares, "square"), (self.players, "player"), (self.coins, "coin"), (self.enemies, "enemy")]:
            for thing in [self.squares, self.coins, self.players, self.enemies]: #this controls the layer order
                for position in thing[1]:
                    #print("blitting ", desc, " @ ", position, sep='')
                    self.screen.blit(thing[0], position)
            
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
        if type(world) is MazeModel:
            self.players[1].clear()
            self.players[1].append((world.PLAYER_AVATAR.coords[1]*self.xSize, world.PLAYER_AVATAR.coords[0]*self.ySize))
            
            self.coins[1].clear()
            for viewModel, entities in [(self.coins, world.coins)]:
                for entity in entities:
                    y,x = entity.coords
                    viewModel[1].append((x*self.xSize, y*self.ySize))
class MazeEnv():
    def reset(self,seed:int=None) -> None:
        return self.model.reset(seed)
    def __init__(self, render_mode : (None|str)=None, nCoins:int=3, startPosition:(None|tuple[int,int])=None) -> None:
        """
        Initialize a new TagEnv.

        Args.
        render_mode: The environment will launch with a human-readable GUI if render_mode is exactly "human".
        nCoins: Number of coins present at each step. Lower values increase the environment difficulty.
        startPosition: Initial coordinates of agent at each epoch, or None for random initial coordinates. Setting to random increases the environment difficulty.
        """
        self.model = MazeModel(
            squares = [ #contents of each square
                [Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY],
                [Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY],
                [Square.EMPTY, Square.EMPTY, Square.SOLID, Square.SOLID, Square.EMPTY, Square.EMPTY],
                [Square.EMPTY, Square.EMPTY, Square.SOLID, Square.SOLID, Square.EMPTY, Square.EMPTY],
                [Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY],
                [Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY],
            ],
            startPosition=startPosition,
            nCoins=nCoins,
            gameLength=100
        )
        if (render_mode=="human"):
            self.view = MazeView(resolution=(500,500), world=self.model)
            self.model.addObserver(self.view)
            self.model.notify()
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
    def validActions(self,s):
        return [0,1,2,3,4]