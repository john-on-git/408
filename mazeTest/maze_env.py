#template from https://www.pygame.org/docs/
import pygame
from enum import Enum
from threading import Thread
from observer import Observable, Observer
import random
import sys

#models
class Square(Enum):
    EMPTY = 0,
    SOLID = 1
class Action(Enum):
    UP    = 0,
    LEFT  = 1,
    DOWN  = 2,
    RIGHT = 3,
    PASS  = 4

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
    
    def __init__(self, squares : [[Square]], playerAvatar : Mover, nCoins : int, gameLength:int=None, scorePerCoin:int=1) -> None:
        super().__init__()
        #assert shape is ok (all rows must be the same length)
        WIDTH = len(squares[0])
        for row in squares:
            assert len(row) == WIDTH
        #coupling++ here but idk if it's worth fixing
        self.SQUARES = squares
        self.COINS = []
        self.ENEMIES = []
        self.PLAYER_AVATAR = playerAvatar
        self.time = 0
        self.GAME_LENGTH = gameLength
        self.score = 0
        self.SCORE_PER_COIN = scorePerCoin

        self.emptySquares = [] #pre-calculated for placing entities
        for x in range(len(self.SQUARES)):
            for y in range(len(self.SQUARES[x])):
                if self.SQUARES[x][y] == Square.EMPTY:
                    self.emptySquares.append((x,y))

        for _ in range(nCoins): #add coins
            self.placeCoin()
    def step(self, action : Action=Action.PASS) -> None:
        def canMoveTo(coords):
            y,x = coords
            return x>0 and y>0 and y<len(self.SQUARES) and x<len(self.SQUARES[0]) and self.SQUARES[y][x] == Square.EMPTY
        
        self.time+=1 #advance time
        if self.GAME_LENGTH is None or self.time<self.GAME_LENGTH:
            markedForDelete = [] #things to be deleted this step
            #move player avatar
            match action:
                case Action.UP:
                    target = (self.PLAYER_AVATAR.coords[0]-1, self.PLAYER_AVATAR.coords[1])
                case Action.LEFT:
                    target = (self.PLAYER_AVATAR.coords[0], self.PLAYER_AVATAR.coords[1]-1)
                case Action.DOWN:
                    target = (self.PLAYER_AVATAR.coords[0]+1, self.PLAYER_AVATAR.coords[1])
                case Action.RIGHT:
                    target = (self.PLAYER_AVATAR.coords[0], self.PLAYER_AVATAR.coords[1]+1)
                case Action.PASS:
                    return
                
            if canMoveTo(target):
                self.PLAYER_AVATAR.coords = target
            #coin collection, spawn new coin
                for coin in self.COINS:
                    if coin.coords == self.PLAYER_AVATAR.coords:
                        markedForDelete.append(coin)
                        self.score+=self.SCORE_PER_COIN
            #enemy movement
            #check for loss
                         
            for entity in markedForDelete:
                if type(entity) == Coin:
                    self.COINS.remove(entity)
                    self.placeCoin() #remember to replace it
                #self.enemies.remove(entity)
        
        self.notify()
        
        percept = None #TODO
        reward = self.score
        terminated = self.GAME_LENGTH is not None and (self.time>=self.GAME_LENGTH) #end of game because of time out
        truncated = None #end of game because the player died
        return (percept, reward, terminated, truncated)
    def placeCoin(self) -> None:
        emptySquares = self.emptySquares.copy()

        emptySquares.remove(self.PLAYER_AVATAR.coords)
        for xs in [self.COINS, self.ENEMIES]:
            for x in xs:
                emptySquares.remove(x.coords)

        self.COINS.append(Coin(coords=random.choice(emptySquares)))
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
        self.players[0] = pygame.Surface((self.xSize, self.ySize))
        playerImage = pygame.transform.scale(pygame.image.load("greenPersonSolid.png"), (self.xSize, self.ySize)) #TODO transparency doesn't work, probably a file format thing
        self.players[0].blit(playerImage, (0,0))

        #coin surface
        self.coins[0] = pygame.Surface((self.xSize, self.ySize))
        coinImage = pygame.transform.scale(pygame.image.load("coinSolid.png"), (self.xSize, self.ySize)) #TODO transparency doesn't work, probably a file format thing
        self.coins[0].blit(coinImage, (0,0))
        
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
        for viewModel, entities in [(self.coins, world.COINS), (self.enemies, world.ENEMIES)]:
            for entity in entities:
                y,x = entity.coords
                viewModel[1].append((x*self.xSize, y*self.ySize))

        self.clock = pygame.time.Clock()
        self.running = False
    def main(self):
        while self.running:
            # pygame.QUIT event means the user clicked X to close your window
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

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
        pygame.quit()
    def open(self):
        self.running = True
        thread = Thread(target=self.main, daemon=True)
        thread.start()
    def close(self):
        self.running = False
    def update(self, world):
        if type(world) is MazeModel:
            self.players[1].clear()
            self.players[1].append((world.PLAYER_AVATAR.coords[1]*self.xSize, world.PLAYER_AVATAR.coords[0]*self.ySize))
            
            self.coins[1].clear()
            for viewModel, entities in [(self.coins, world.COINS), (self.enemies, world.ENEMIES)]:
                for entity in entities:
                    y,x = entity.coords
                    viewModel[1].append((x*self.xSize, y*self.ySize))
class MazeEnv():
    def reset(self) -> None:
        self.model.SQUARES = [ #contents of each square
            [Square.SOLID, Square.SOLID, Square.SOLID, Square.SOLID, Square.SOLID],
            [Square.SOLID, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.SOLID],
            [Square.SOLID, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.SOLID],
            [Square.SOLID, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.SOLID],
            [Square.SOLID, Square.SOLID, Square.SOLID, Square.SOLID, Square.SOLID],
        ]
        self.model.PLAYER_AVATAR=Mover(coords=(2,2))
    def __init__(self, renderMode : str) -> None:
        self.humanRender = (renderMode=="human")
        self.model = MazeModel(
            squares = [ #contents of each square
                [Square.SOLID, Square.SOLID, Square.SOLID, Square.SOLID, Square.SOLID],
                [Square.SOLID, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.SOLID],
                [Square.SOLID, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.SOLID],
                [Square.SOLID, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.SOLID],
                [Square.SOLID, Square.SOLID, Square.SOLID, Square.EMPTY, Square.SOLID],
                [Square.SOLID, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.SOLID],
                [Square.SOLID, Square.EMPTY, Square.SOLID, Square.SOLID, Square.SOLID],
                [Square.SOLID, Square.EMPTY, Square.SOLID, Square.EMPTY, Square.SOLID],
                [Square.SOLID, Square.EMPTY, Square.SOLID, Square.EMPTY, Square.SOLID],
                [Square.SOLID, Square.EMPTY, Square.SOLID, Square.EMPTY, Square.SOLID],
                [Square.SOLID, Square.EMPTY, Square.EMPTY, Square.EMPTY, Square.SOLID],
                [Square.SOLID, Square.SOLID, Square.SOLID, Square.SOLID, Square.SOLID],
            ],
            playerAvatar=Mover(coords=(2,2)),
            nCoins=3
        )
        if self.humanRender:
            self.view = MazeView(resolution=(500,500), world=self.model)
            self.model.addObserver(self.view)
            self.view.open()
    def step(self, action):
        return self.model.step(action)
    def close(self):
        self.view.close()