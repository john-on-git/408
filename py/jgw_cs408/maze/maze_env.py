#template from https://www.pygame.org/docs/
import pygame
from enum import Enum
from threading import Thread
from jgw_cs408.observer import Observable, Observer
import random

#models
class Square(Enum):
    EMPTY = 0,
    SOLID = 1

class Entity():
    def __init__(self, coords) -> None:
        self.coords = coords
class Mover(Entity):
    def __init__(self, coords) -> None:
        super().__init__(coords)
class Coin(Entity):
    def __init__(self, coords) -> None:
        super().__init__(coords)

class MazeModel(Observable):
    def __init__(self, squares : list[list[Square]], playerPosition : (str|tuple), nCoins : int, gameLength:int=None, scorePerCoin:int=10) -> None:
        super().__init__()
        #assert shape is ok (all rows must be the same length)
        WIDTH = len(squares[0])
        for row in squares:
            assert len(row) == WIDTH
        #coupling++ here but idk if it's worth fixing
        self.squares = squares
        self.emptySquares = [] #pre-calculated for placing entities
        for y in range(len(self.squares)):
            for x in range(len(self.squares[y])):
                if self.squares[y][x] == Square.EMPTY:
                    self.emptySquares.append((y,x))

        self.coins = []
        self.enemies = []
        if playerPosition=="random":
            self.placePlayer()
        else:
            self.PLAYER_AVATAR = Mover(playerPosition)
            
        self.time = 0
        self.GAME_LENGTH = gameLength
        self.score = 0
        self.scorePerCoin = scorePerCoin
        self.food = self.scorePerCoin

        for _ in range(nCoins): #add coins
            self.placeCoin()
    def reset(self, squares : list[list[Square]], playerPosition : (str|tuple), nCoins : int) -> None:
        self.squares = squares
        if playerPosition=="random":
            self.placePlayer()
        else:
            self.PLAYER_AVATAR = Mover(playerPosition)
        self.coins.clear()
        self.score = 0
        self.time = 0
        self.food = self.scorePerCoin
        for _ in range(nCoins): #add coins
            self.placeCoin()
    def step(self, action : (0|1|2|3|4) = 4) -> tuple[list[float], int, bool, bool, None]:
        def canMoveTo(coords : tuple) -> bool:
            y,x = coords
            return x>=0 and y>=0 and y<len(self.squares) and x<len(self.squares[0]) and self.squares[y][x] == Square.EMPTY

        self.food-=1
        reward = 0
        self.time+=1 #advance time
        if self.GAME_LENGTH is None or self.time<self.GAME_LENGTH:
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
                        reward+=self.scorePerCoin
                        self.score+=self.scorePerCoin
            #enemy movement
            #check for loss
                         
            for entity in markedForDelete:
                if type(entity) == Coin:
                    self.coins.remove(entity)
                    self.placeCoin() #remember to replace it
                    self.food+=self.scorePerCoin
                #self.enemies.remove(entity)
        
        self.notify()
        
        #reward = reward
        logits = self.calcLogits()
        terminated = self.GAME_LENGTH is not None and (self.time>=self.GAME_LENGTH) #end of game because of time out
        truncated = self.food<=0 #end of game because the player died
        info = None
        return (logits, reward, terminated, truncated, info)
    def calcLogits(self) -> list[float]:
        LOGIT_EMPTY  = 0.0
        LOGIT_SOLID  = 1.0
        LOGIT_PLAYER = 2.0
        LOGIT_COIN   = 3.0
        LOGIT_ENEMY  = 4.0
        def calcLogit(y: int,x : int) -> float:
            if self.squares[y][x] == Square.SOLID:
                return LOGIT_SOLID
            elif self.PLAYER_AVATAR.coords == (y,x):
                return LOGIT_PLAYER
            else:
                for coin in self.coins:
                    if coin.coords == (y,x):
                        return LOGIT_COIN
                for enemy in self.enemies:
                    if enemy.coords == (y,x):
                        return LOGIT_ENEMY
                return LOGIT_EMPTY
        #construct logits from world
        logits = []
        for y in range(len(self.squares)):
            logits.append([None] * len(self.squares))
            for x in range(len(logits[0])):
                logits[y][x] = calcLogit(y,x)
        return logits
    def placePlayer(self) -> tuple:
        emptySquares = self.emptySquares.copy()
        for xs in [self.coins, self.enemies]:
            for x in xs:
                emptySquares.remove(x.coords)
        self.PLAYER_AVATAR = Mover(coords=random.choice(emptySquares))
    def placeCoin(self) -> None:
        emptySquares = self.emptySquares.copy()
        emptySquares.remove(self.PLAYER_AVATAR.coords)
        for xs in [self.coins, self.enemies]:
            for x in xs:
                emptySquares.remove(x.coords)
        self.coins.append(Coin(coords=random.choice(emptySquares)))
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
        self.xSize = displayResolution[0]/len(world.squares[0]) #horizontal size of squares
        self.ySize = displayResolution[1]/len(world.squares) #vertical size of squares
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
        for y in range(len(world.squares)):
            for x in range(len(world.squares[y])):
                if(world.squares[y][x] == Square.EMPTY):
                    self.squares[1].append((x*self.xSize, y*self.ySize))

        #get player location
        self.players[1].clear()
        self.players[1].append((world.PLAYER_AVATAR.coords[1]*self.xSize, world.PLAYER_AVATAR.coords[0]*self.ySize))
        
        #get coin locations
        self.coins[1].clear()
        for viewModel, entities in [(self.coins, world.coins), (self.enemies, world.enemies)]:
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
            for viewModel, entities in [(self.coins, world.coins), (self.enemies, world.enemies)]:
                for entity in entities:
                    y,x = entity.coords
                    viewModel[1].append((x*self.xSize, y*self.ySize))
class MazeEnv():
    def reset(self,seed:int=None) -> None:
        self.model.reset(
            squares = [ #contents of each square
                [Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY],
                [Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY],
                [Square.EMPTY, Square.EMPTY, Square.SOLID, Square.SOLID, Square.EMPTY, Square.EMPTY],
                [Square.EMPTY, Square.EMPTY, Square.SOLID, Square.SOLID, Square.EMPTY, Square.EMPTY],
                [Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY],
                [Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY],
            ],
            playerPosition=self.startPosition,
            nCoins=self.nCoins
        )
        return (self.model.calcLogits(), None)
    def __init__(self, nCoins:int=3, startPosition:(str|tuple)="random", render_mode : (None|str)=None) -> None:
        self.actionSpace = [0,1,2,3,4]
        self.nCoins=nCoins
        self.startPosition = startPosition
        self.model = MazeModel(
            squares = [ #contents of each square
                [Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY],
                [Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY],
                [Square.EMPTY, Square.EMPTY, Square.SOLID, Square.SOLID, Square.EMPTY, Square.EMPTY],
                [Square.EMPTY, Square.EMPTY, Square.SOLID, Square.SOLID, Square.EMPTY, Square.EMPTY],
                [Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY],
                [Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.EMPTY],
            ],
            playerPosition=self.startPosition,
            nCoins=self.nCoins,
            gameLength=100
        )
        if (render_mode=="human"):
            self.view = MazeView(resolution=(500,500), world=self.model)
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
    def validActions(self,s):
        return self.actionSpace