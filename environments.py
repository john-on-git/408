from abc import ABC, abstractmethod
from enum import Enum
from observer import Observable, Observer
import math
import pygame
from random import Random
import tensorflow as tf
from threading import Thread

class Environment(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.ACTION_SPACE: list
        self.View: View | None 
        self.terminated: bool
        self.truncated: bool
    def close(self) -> None:
        if self.view is not None:
            self.view.close()
            self.view = None
        self.terminated = True
    @abstractmethod
    def reset(self, seed:int) -> tuple[list[float], dict]:
        pass
    @abstractmethod
    def step(self,a:int) -> tuple[list[float], int, bool, bool, dict]:
        pass
    @abstractmethod
    def validActions(self,s) -> list[int]:
        pass
class View(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.running: bool
        self.thread: Thread
    def open(self):
        self.running = True
        self.thread = Thread(target=self.main, daemon=True)
        self.thread.start()
    def close(self):
        self.running = False
        self.thread.join()
        pygame.quit()
    @abstractmethod
    def main(self):
        pass
    @abstractmethod
    def update(self, world):
        pass

#An extremely simple environment for testing agents.
#If an agent can't achieve the optimal policy on this, it indicates that there's something wrong with the implementation. 
class TestBanditEnv(Environment):
    def __init__(self,nMachines:int=2) -> None:
        super().__init__()
        self.ACTION_SPACE = range(nMachines)
    def reset(self, seed:int=None) -> tuple[list[float], dict]:
        return ([1], {})
    def step(self, a: int) -> tuple[list[float], int, bool, bool, dict]:
        reward = 1 if a==0 else 0
        return ([1], reward, True, False, {})
    def validActions(self, s) -> list[int]:
        return self.ACTION_SPACE
#Maze
MAZE_REWARD_PER_COIN = 50
MAZE_REWARD_EXPLORATION = 1

#models
class MazeSquare(Enum):
    EMPTY = 0,
    SOLID = 1
class MazeEntity():
    def __init__(self, coords) -> None:
        self.coords = coords
class MazeCoin(MazeEntity):
    def __init__(self, coords) -> None:
        super().__init__(coords)
class MazeEnv(Environment, Observable):
    def __init__(self, render_mode:(None|str)=None, startPosition:(str|tuple)="random", nCoins:int=1, gameLength:int=50, squares=None, rewardExploration:bool=True) -> None:
        """
        Initialize a new MazeEnv.

        Args.
        render_mode: The environment will launch with a human-readable GUI if render_mode is exactly "human".
        nCoins: Number of coins present at each step. Higher values increase the environment difficulty (by enlarging the state space).
        startPosition: Initial coordinates of agent at each epoch, or None for random initial coordinates. Setting to random increases the environment difficulty.
        gameLength: Max length of game, or None for unlimited length.
        """
        Observable.__init__(self)
        #init constants
        self.ACTION_SPACE = [0,1,2,3,4]
        self.SQUARES = squares if squares is not None else [ #TODO should not be hard-coded
            [MazeSquare.EMPTY, MazeSquare.EMPTY, MazeSquare.EMPTY, MazeSquare.EMPTY, MazeSquare.EMPTY, MazeSquare.EMPTY],
            [MazeSquare.EMPTY, MazeSquare.EMPTY, MazeSquare.EMPTY, MazeSquare.EMPTY, MazeSquare.EMPTY, MazeSquare.EMPTY],
            [MazeSquare.EMPTY, MazeSquare.EMPTY, MazeSquare.SOLID, MazeSquare.SOLID, MazeSquare.EMPTY, MazeSquare.EMPTY],
            [MazeSquare.EMPTY, MazeSquare.EMPTY, MazeSquare.SOLID, MazeSquare.SOLID, MazeSquare.EMPTY, MazeSquare.EMPTY],
            [MazeSquare.EMPTY, MazeSquare.EMPTY, MazeSquare.EMPTY, MazeSquare.EMPTY, MazeSquare.EMPTY, MazeSquare.EMPTY],
            [MazeSquare.EMPTY, MazeSquare.EMPTY, MazeSquare.EMPTY, MazeSquare.EMPTY, MazeSquare.EMPTY, MazeSquare.EMPTY],
        ]
        self.INITIAL_PLAYER_POSITION = startPosition
        self.GAME_LENGTH = gameLength
        self.N_COINS = nCoins
        self.REWARD_EXPLORATION = rewardExploration
        self.EMPTY_SQUARES = [] #pre-calculated for placing entities
        for y in range(len(self.SQUARES)):
            for x in range(len(self.SQUARES[y])):
                if self.SQUARES[y][x] == MazeSquare.EMPTY:
                    self.EMPTY_SQUARES.append((y,x))
        self.reset()
        if (render_mode=="human"):
            self.view = MazeView(resolution=(500,500), world=self)
            self.addObserver(self.view)
            self.view.open()
        else:
            self.view = None
    def reset(self, seed:int=None) -> None:
        self.random = Random(seed)
        self.time = 0
        self.terminated = False
        self.truncated = False
        self.coins = []
        self.visited = []
        if self.INITIAL_PLAYER_POSITION=="random":
            emptySquares = self.EMPTY_SQUARES.copy()
            for xs in [self.coins]:
                for x in xs:
                    emptySquares.remove(x.coords)
            self.PLAYER_AVATAR = MazeEntity(coords=self.random.choice(emptySquares))
        else:
            self.PLAYER_AVATAR = MazeEntity(self.INITIAL_PLAYER_POSITION)
        self.visited.append(self.PLAYER_AVATAR.coords)
        for _ in range(self.N_COINS): #add coins
            self.placeCoin()
        self.notify() #update view
        return (self.calcLogits(), {})
    def step(self, action:(0|1|2|3|4)):
        reward = 0
        markedForDelete = [] #things to be deleted this step
        if not self.terminated and not self.truncated:
            self.time+=1 #update time
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
            #move the player to the target if possible (it must be an empty square inside the game board)
            y,x = target
            if x>=0 and y>=0 and y<len(self.SQUARES) and x<len(self.SQUARES[0]) and self.SQUARES[y][x] == MazeSquare.EMPTY:
                self.PLAYER_AVATAR.coords = target
            if self.REWARD_EXPLORATION and self.PLAYER_AVATAR.coords not in self.visited:
                self.visited.append(self.PLAYER_AVATAR.coords)
                reward+=MAZE_REWARD_EXPLORATION
            #coin collection, spawn new coin
            for coin in self.coins:
                if coin.coords == self.PLAYER_AVATAR.coords:
                    markedForDelete.append(coin)
                    reward+=MAZE_REWARD_PER_COIN
            #check for loss
            if self.GAME_LENGTH is not None and self.time>=self.GAME_LENGTH:
                self.terminated = True #end of game because of time out
        #reward = reward
        logits = self.calcLogits()
        info = {}
        
        for entity in markedForDelete:
            if type(entity) == MazeCoin:
                self.coins.remove(entity)
                if len(self.coins)<self.N_COINS: #always true during normal gameplay, added to make this possible to unit test. See what I mean?
                    self.placeCoin() #remember to replace it
        self.notify() #update view
        return (logits, reward, self.terminated, self.truncated, info)
    def placeCoin(self) -> None:
        validForCoin = self.EMPTY_SQUARES.copy() #list of empty squares
        validForCoin.remove(self.PLAYER_AVATAR.coords) #can't place on player
        for xs in [self.coins]: #or on top of another coin
            for x in xs:
                validForCoin.remove(x.coords)
        self.coins.append(MazeCoin(coords=self.random.choice(validForCoin)))  
    def validActions(self,s):
        return [0,1,2,3] #AI agents can't pass as it's never the optimal move
    def notify(self) -> None:
        observer: Observer
        for observer in super().getObservers():
            observer.update(self)
    def calcLogits(self) -> list[float]:
        LOGIT_EMPTY  = 0.0
        LOGIT_SOLID  = 1.0
        LOGIT_COIN   = 2.0
        LOGIT_PLAYER = 4.0
        #construct logits from world
        logits = []
        for y in range(len(self.SQUARES)):
            logits.append([LOGIT_SOLID] * len(self.SQUARES))
        for y,x in self.EMPTY_SQUARES:
            logits[y][x] = LOGIT_EMPTY
        for coin in self.coins:
            y,x = coin.coords
            logits[y][x] += LOGIT_COIN
        y,x = self.PLAYER_AVATAR.coords
        logits[y][x] += LOGIT_PLAYER
        return logits
#view
class MazeView(View, Observer):
    def __init__(self, world : MazeEnv, resolution) -> None:
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
        self.players[0] = pygame.transform.scale(pygame.image.load("img/greenPerson.png"), (self.xSize, self.ySize)) #TODO transparency doesn't work, probably a file format thing

        #coin surface
        self.coins[0] = pygame.transform.scale(pygame.image.load("img/coin.png"), (self.xSize, self.ySize)) #TODO transparency doesn't work, probably a file format thing
        
        #create squares
        for y in range(len(world.SQUARES)):
            for x in range(len(world.SQUARES[y])):
                if(world.SQUARES[y][x] == MazeSquare.EMPTY):
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
            #template from https://www.pygame.org/docs/
            #fill the screen with a color to wipe away anything from last frame
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
    def update(self, world):
        if type(world) is MazeEnv:
            self.players[1].clear()
            self.players[1].append((world.PLAYER_AVATAR.coords[1]*self.xSize, world.PLAYER_AVATAR.coords[0]*self.ySize))
            
            self.coins[1].clear()
            for viewModel, entities in [(self.coins, world.coins)]:
                for entity in entities:
                    y,x = entity.coords
                    viewModel[1].append((x*self.xSize, y*self.ySize))



#Tag
TAG_REWARD_PER_STEP = 1

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
class TagEnv(Environment, Observable):
    def __init__(self, render_mode:(None|str)=None, maxTime=1000, nSeekers = 1,  speedRatio = 2/3, seekerSpread = 180, seekerMinmaxDistance=(100,200), arenaDimensions=(500,500)) -> None:
        """
        Initialize a new TagEnv.

        Args.
        render_mode: The environment will launch with a human-readable GUI if render_mode is exactly "human".
        maxTime: Max number of steps until epoch termination.
        nSeekers: Number of hostile agents. Higher values increase the environment difficulty.
        speedRatio: Ratio of speed of player avatar relative to hostile agents. Higher values increase the environment difficulty.
        seekerSpread: Max deflection away from 180 degreees behind relative at which the the seekers can spawn. Higher values increase the environment difficulty
        seekerMinMaxDistance: Min and max distances from the runner at which the seekers can spawn.
        arenaDimensions: Dimensions of the game arena. Lower values increase the environmental difficulty.
        """
        
        Observable.__init__(self)
        #init constants
        self.ACTION_SPACE = [0,1,2]
        self.SCALE = 10
        self.MAX_TIME = maxTime
        RUNNER_SPEED = 5
        RUNNER_ROTATION_RATE = math.pi/30
        #load hitbox masks
        RUNNER_HITBOX_FACTORY       = pygame.transform.scale_by(pygame.image.load("img/runner.png"), self.SCALE).get_rect
        self.SEEKER_HITBOX_FACTORY  = pygame.transform.scale_by(pygame.image.load("img/seeker.png"), self.SCALE).get_rect

        arenaX, arenaY = arenaDimensions
        self.seekerMinDistance, self.seekerMaxDistance = seekerMinmaxDistance
        #parameters to reset to when RaceModel.reset() is called
        self.RUNNER_INITIAL_POSITION = (arenaX/2 * self.SCALE, arenaY/2 * self.SCALE)
        self.RUNNER_INITIAL_ROTATION = (lambda: self.random.random()*2*math.pi)
        self.N_SEEKERS = nSeekers
        self.SEEKER_SPEED = RUNNER_SPEED * speedRatio
        self.SEEKER_SPREAD = math.radians(seekerSpread)

        #init entities
        self.ARENA = Entity(pygame.Rect(0,0,arenaX*self.SCALE,arenaY*self.SCALE), 0) #game ends if agent is not in contact with this rect
        self.RUNNER = Mover(rect=RUNNER_HITBOX_FACTORY(), rotation=0, speed=RUNNER_SPEED * self.SCALE, rotationRate=RUNNER_ROTATION_RATE)
        self.RUNNERS = [self.RUNNER]
        self.SEEKERS = []

        self.reset(None)
        
        if (render_mode=="human"):
            self.view = TagView(resolution=pygame.Vector2(750,500), model=self)
            self.addObserver(self.view)
            self.notify()
            self.view.open()
        else:
            self.view = None
    def reset(self, seed:int=None) -> tuple[list[float], int, bool, bool, None]:
        def genSeekerPosition(dist=None):
            dist = self.random.randint(self.seekerMinDistance*self.SCALE,self.seekerMaxDistance*self.SCALE) if dist==None else dist
            x,y = self.RUNNER.rect.center
            angle = self.RUNNER.rotation + math.pi + (self.random.random() * self.SEEKER_SPREAD) - (self.SEEKER_SPREAD/2) #180 degree cone behind runner 
            x += math.cos(angle)*dist
            y += math.sin(angle)*dist
            return (x,y)
        self.random = Random(seed)
        self.RUNNER.rect.center = self.RUNNER_INITIAL_POSITION
        self.RUNNER.rotation = self.RUNNER_INITIAL_ROTATION()

        #reset seekers
        self.SEEKERS.clear()
        for _ in range(self.N_SEEKERS):
            seeker = Mover(rect=self.SEEKER_HITBOX_FACTORY(), rotation=0, speed=self.SEEKER_SPEED * self.SCALE)
            seeker.rect.center = genSeekerPosition()
            self.SEEKERS.append(seeker)

        self.terminated = False
        self.truncated = False
        self.time = 0
        self.notify() #redraw
        return (self.calcLogits(), {}) 
    def step(self, action : (0|1|2) = 1) -> tuple[tuple, int, bool, bool, dict]:
        reward = TAG_REWARD_PER_STEP #baseline per step
        if not self.terminated and not self.truncated:
            self.time+=1
            #update the runner's angle according to the action
            match action:
                case 0: #turn left
                    self.RUNNER.left()
                case 1: #no action
                    pass
                case 2: #turn right
                    self.RUNNER.right()
                case _:
                    raise ValueError() if type(action) is int else TypeError()
            #update the runner's position
            self.RUNNER.advance()
            #update the seekers' angles & positions
            for seeker in self.SEEKERS:
                dx = self.RUNNER.rect.center[0] - seeker.rect.center[0]
                dy = self.RUNNER.rect.center[1] - seeker.rect.center[1]
                seeker.rotation = math.atan2(dy, dx)
                seeker.advance()
                
            if (self.RUNNER.rect.collidelist([seeker.rect for seeker in self.SEEKERS]) != -1) or (not self.RUNNER.rect.colliderect(self.ARENA)): #check for collisions
                self.truncated = True
            elif self.time==self.MAX_TIME: #and time out
                self.terminated = True
        logits = self.calcLogits()
        info = {}
        self.notify() #update view
        return (logits, reward, self.terminated, self.truncated, info)
    def validActions(self,s):
        return self.ACTION_SPACE
    def notify(self) -> None:
        observer: Observer
        for observer in super().getObservers():
            observer.update(self)
    def calcLogits(self) -> list[float]:
        x = [float(self.RUNNER.rect.center[0]), float(self.RUNNER.rect.center[1]), self.RUNNER.rotation]
        for seeker in self.SEEKERS:
            x.extend([float(seeker.rect.center[0]), float(seeker.rect.center[1])])
        return x
#view
class TagView(View, Observer):
    def __init__(self, model : TagEnv, resolution) -> None:
        super().__init__()
        self.BG_COLOR = pygame.color.Color(126,126,126)
        pygame.init()
        displayResolution = resolution
        self.screen = pygame.display.set_mode(displayResolution)
        
        #TODO
        self.scale = 1/model.SCALE

        #load images
        runnerSurface = pygame.image.load("img/runner.png")
        seekerSurface = pygame.image.load("img/seeker.png")
        arenaSurface  = pygame.Surface(pygame.Vector2(model.ARENA.rect.size) * self.scale)
        arenaSurface.fill((255,255,255))

        #init drawing data structures
        self.cameraPos = model.RUNNER.getCenter
        self.centerPos = pygame.Vector2(resolution[0]/2, resolution[1]/2)

        self.runners = []
        self.seekers = []
        self.arenas = [model.ARENA]
        self.drawTargets = [(arenaSurface, self.arenas),(seekerSurface, self.seekers), (runnerSurface, self.runners)] #order controls the layering
        
        self.clock = pygame.time.Clock()
        self.running = False
    def main(self):
        while self.running:
            #template from https://www.pygame.org/docs/
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
    def update(self, world):
        if type(world) is TagEnv:
            for _, xs in self.drawTargets:
                xs.clear()
            for runner in world.RUNNERS:
                self.runners.append(runner)
            for seeker in world.SEEKERS:
                self.seekers.append(seeker)
            self.arenas.append(world.ARENA)



#Tic-Tac-Toe
TTT_REWARD_PARTIAL_CHAIN = 2
TTT_REWARD_WIN = 10
TTT_REWARD_PER_STEP = 1
TTT_REWARD_INVALID = -100

#models
class TTTSearchAgent(): #agent that follows the optimal policy by performing a tree search
    def  __init__(self, random:Random, epsilon=0, epsilonDecay=1, depth=-1) -> None:
        self.epsilon=epsilon
        self.epsilonDecay = epsilonDecay
        self.depth=depth
        self.random = random
    def act(self, s) -> int:
        #0.0 = Empty
        #1.0 = Player
        #2.0 = Enemy
        def actionSpace(s):
            actions = []
            for i in range(len(s[0])):
                if s[0][i]==0.0:
                    actions.append(i)
            return actions
        def succ(s, a, actor):
            s = s.copy()
            s[0][a] = actor
            return s
        def isTerminalAndValue(s) -> (int|None): #returns the value if s is terminal, False if it is not
            def calcLongestChains(board) -> dict[float, int]:
                longestChains = {1.0:0, 2.0:0}
                #check horizontals
                for i in range(n):
                    chain = None
                    for j in range(n):
                        if chain==None:
                            if board[i][j] != 0.0: #chain starts
                                chain = [board[i][j], 1]
                                if longestChains[chain[0]]<chain[1]:
                                    longestChains[chain[0]] = chain[1]
                        else:
                            if board[i][j] == chain[0]: #chain continues
                                chain[1]+=1
                                if longestChains[chain[0]]<chain[1]:
                                    longestChains[chain[0]] = chain[1]
                            else: #chain ends
                                chain = None
                #check verticals
                for j in range(n):
                    chain = None
                    for i in range(n):
                        if chain==None:
                            if board[i][j] != 0.0: #chain starts
                                chain = [board[i][j], 1]
                                if longestChains[chain[0]]<chain[1]:
                                    longestChains[chain[0]] = chain[1]
                        else:
                            if board[i][j] == chain[0]: #chain continues
                                chain[1]+=1
                                if longestChains[chain[0]]<chain[1]:
                                    longestChains[chain[0]] = chain[1]
                            else: #chain ends
                                chain = None
                #check diagonals
                #\
                chain = None
                for i in range(n):
                    if chain==None:
                        if board[i][i] != 0.0: #chain starts
                            chain = [board[i][i], 1]
                            if longestChains[chain[0]]<chain[1]:
                                longestChains[chain[0]] = chain[1]
                    else:
                        if board[i][i] == chain[0]: #chain continues
                            chain[1]+=1
                            if longestChains[chain[0]]<chain[1]:
                                longestChains[chain[0]] = chain[1]
                        else: #chain ends
                            chain = None
                #/
                chain = None
                for i in range(n):
                        if chain==None:
                            if board[i][n-1-i] != 0.0: #chain starts
                                chain = [board[i][n-1-i], 1]
                                if longestChains[chain[0]]<chain[1]:
                                    longestChains[chain[0]] = chain[1]
                        else:
                            if board[i][n-1-i] == chain[0]: #chain continues
                                chain[1]+=1
                                if longestChains[chain[0]]<chain[1]:
                                    longestChains[chain[0]] = chain[1]
                            else: #chain ends
                                chain = None

                return longestChains
            
            #reconstruct board from logits
            n = int(math.sqrt(len(s[0])))
            board = []
            for i in range(n):
                board.append([])
                for j in range(n):
                    board[i].append(s[0][n*i+j])
            longestChains = calcLongestChains(board)

            if longestChains[1.0]==n:
                return 999
            elif longestChains[2.0]==n:
                return -999
            else:
                res = 0
                for col in board:
                    for cell in col:
                        if cell == 0.0:
                            res = None
                return res
        def mini(s, depth, alpha, beta) -> int:
            if depth == 0:
                return 0
            else:
                tV = isTerminalAndValue(s)
                if tV==None: #recurse
                    minScore = float('inf')
                    for a in actionSpace(s):
                        minScore = min(minScore, maxi(succ(s,a,2.0), depth-1, alpha, beta))
                        if minScore<alpha or minScore==-999:
                            break
                        beta = min(beta,minScore)
                    return minScore-1 #-1 time penalty
                else: #stop if terminal
                    return tV
        def maxi(s, depth, alpha, beta) -> int:
            if depth == 0:
                return 0
            else:
                tV = isTerminalAndValue(s)
                if tV==None: #recurse
                    maxScore = -float('inf')
                    for a in actionSpace(s):
                        maxScore = max(maxScore, mini(succ(s,a,1.0), depth-1, alpha, beta))
                        if maxScore>beta or maxScore==999:
                            break
                        alpha = max(alpha,maxScore)
                    return maxScore-1 #-1 time penalty
                else: #stop if terminal
                    return tV
        self.epsilon *= self.epsilonDecay
        if self.random.random()<self.epsilon: #chance to act randomly
            valid = []
            for i in range(9):
                if s[0][i] == 0.0:
                    valid.append(i)
            return self.random.choice(valid)
        else:
            s = s.numpy()
            maxScore = (None, -float('inf'))
            for a in actionSpace(s):
                score = mini(succ(s,a,1.0), self.depth-1, -float('inf'), float('inf')) 
                if score==999:
                    return a
                elif score>maxScore[1]:
                    maxScore = (a, score)
            return maxScore[0]
class Team(Enum):
    EMPTY = 0.0,
    NOUGHT = 1.0,
    CROSS = 2.0
class TTTEnv (Environment, Observable):
    def __init__(self, render_mode:(None|str)=None, size:int=3, opponent=TTTSearchAgent(None, .75)) -> None:
        """
        Initialize a new TTTEnv.

        Args.
        render_mode: The environment will launch with a human-readable GUI if render_mode is exactly "human".
        size: Equal to the width and height of the game board.
        opponent: Agent responsible for the enemy AI. The provided agent be guaranteed to provide a valid action for all game states.
        """
        
        Observable.__init__(self)
        self.OPPONENT = opponent
        self.SIZE = size
        self.OPPONENT = opponent
        self.ACTION_SPACE = range(self.SIZE**2)
        self.terminated = False
        self.truncated = False
        self.board = []
        for i in range(self.SIZE):
            self.board.append([])
            for _ in range(self.SIZE):
                self.board[i].append(Team.EMPTY)
        if (render_mode=="human"):
            self.view = TTTView(resolution=(600,600), model=self)
            self.view.open()
        else:
            self.view = None
    def reset(self, seed=None) -> None:
        self.random = Random(seed)
        self.OPPONENT.random = self.random
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                self.board[i][j] = Team.EMPTY
        self.truncated = False
        self.terminated = False
        self.notify() #update view
        return (self.calcLogits(Team.NOUGHT), {})
    def step(self, action): #both sides move
        def halfStep(actor : Team, action : int) -> list[list[float], int, bool, bool, None]: #one side moves
            def calcLongestChains(n) -> dict[Team, int]:
                longestChains = {Team.NOUGHT:0, Team.CROSS:0}
                #check horizontals
                for i in range(n):
                    chain = None
                    for j in range(n):
                        if chain==None:
                            if self.board[i][j] != Team.EMPTY: #chain starts
                                chain = [self.board[i][j], 1]
                                if longestChains[chain[0]]<chain[1]:
                                    longestChains[chain[0]] = chain[1]
                        else:
                            if self.board[i][j] == chain[0]: #chain continues
                                chain[1]+=1
                                if longestChains[chain[0]]<chain[1]:
                                    longestChains[chain[0]] = chain[1]
                            else: #chain ends
                                chain = None
                #check verticals
                for j in range(n):
                    chain = None
                    for i in range(n):
                        if chain==None:
                            if self.board[i][j] != Team.EMPTY: #chain starts
                                chain = [self.board[i][j], 1]
                                if longestChains[chain[0]]<chain[1]:
                                    longestChains[chain[0]] = chain[1]
                        else:
                            if self.board[i][j] == chain[0]: #chain continues
                                chain[1]+=1
                                if longestChains[chain[0]]<chain[1]:
                                    longestChains[chain[0]] = chain[1]
                            else: #chain ends
                                chain = None
                #check diagonals (lines that don't intersect corners don't count as they can't contribute to a winning line)
                #\
                chain = None
                for i in range(n):
                    if chain==None:
                        if self.board[i][i] != Team.EMPTY: #chain starts
                            chain = [self.board[i][i], 1]
                            if longestChains[chain[0]]<chain[1]:
                                longestChains[chain[0]] = chain[1]
                    else:
                        if self.board[i][i] == chain[0]: #chain continues
                            chain[1]+=1
                            if longestChains[chain[0]]<chain[1]:
                                longestChains[chain[0]] = chain[1]
                        else: #chain ends
                            chain = None
                #/
                chain = None
                for i in range(n):
                        if chain==None:
                            if self.board[i][n-1-i] != Team.EMPTY: #chain starts
                                chain = [self.board[i][n-1-i], 1]
                                if longestChains[chain[0]]<chain[1]:
                                    longestChains[chain[0]] = chain[1]
                        else:
                            if self.board[i][n-1-i] == chain[0]: #chain continues
                                chain[1]+=1
                                if longestChains[chain[0]]<chain[1]:
                                    longestChains[chain[0]] = chain[1]
                            else: #chain ends
                                chain = None

                return longestChains
            reward = 0
            actX = action%self.SIZE
            actY = int(action/self.SIZE)
            if not self.terminated and not self.truncated:
                if self.board[actY][actX] == Team.EMPTY: #enact the move if it's valid
                    self.board[actY][actX] = actor
                    longestChains = calcLongestChains(self.SIZE)
                    if longestChains[actor]==self.SIZE: #end the game if there's a winner
                        reward = TTT_REWARD_WIN**self.SIZE #distribute reward for the win
                        self.truncated = True
                    else:
                        #end the game as a draw if all cells are full
                        self.terminated = True
                        for row in self.board:
                            for cell in row:
                                if cell == Team.EMPTY:
                                    self.terminated = False
                        reward += TTT_REWARD_PARTIAL_CHAIN**longestChains[actor] #distribute reward for partial chains
                else:
                    reward = TTT_REWARD_INVALID #distribute negative reward for invalid moves
                reward+=TTT_REWARD_PER_STEP #distribute reward for time
            self.notify() #update view
            return (reward, self.terminated, self.truncated)
        rew, terminated, truncated = halfStep(Team.NOUGHT, action) #handle player action
        if not (terminated or truncated):
            opponentAction = self.OPPONENT.act(tf.expand_dims(tf.convert_to_tensor(self.calcLogits(Team.CROSS)),0))
            if self.board[int(opponentAction/self.SIZE)][opponentAction%self.SIZE] == Team.EMPTY: #enact move if valid
                _, terminated, truncated = halfStep(Team.CROSS, opponentAction) #handle CPU action
                return (self.calcLogits(Team.NOUGHT), rew, terminated, truncated, {})
            else:
                self.terminated = True
                raise Exception("Tic-Tac-Toe opponent made invalid move.")
        else:
            return (self.calcLogits(Team.NOUGHT), rew, terminated, truncated, {}) #last move of the game
    def validActions(self,s):
        valid = []
        for action in self.ACTION_SPACE:
            if s[0][action] == 0.0:
                valid.append(action)
        return valid
    def notify(self) -> None:
        observer: Observer
        for observer in super().getObservers():
            observer.update(self)
    def calcLogits(self, actor :Team) -> list[float]:
            def logit(actor : Team, cell:Team):
                if cell==Team.EMPTY:
                    return 0.0 #empty
                elif cell==actor:
                    return 1.0 #player
                else:
                    return 2.0 #enemy
            logits = []
            for row in self.board:
                for cell in row:
                    logits.append(logit(actor, cell))
            return logits
#view
class TTTView(View, Observer):
    def __init__(self, resolution, model : TTTEnv) -> None:
        super().__init__()
        pygame.init()
        model.addObserver(self)
        displayResolution = resolution
        self.screen = pygame.display.set_mode(displayResolution)
        self.xSize = displayResolution[0]/model.SIZE #horizontal size of board
        self.ySize = displayResolution[1]/model.SIZE #vertical size of board
        self.noughts = [None, []]
        self.crosses = [None, []]

        #nought surface
        self.noughts[0] = pygame.transform.scale(pygame.image.load("img/nought.png"), (self.xSize+1, self.ySize+1)) #TODO transparency doesn't work, probably a file format thing

        #cross surface
        self.crosses[0] = pygame.transform.scale(pygame.image.load("img/cross.png"), (self.xSize+1, self.ySize+1)) #TODO transparency doesn't work, probably a file format thing

        self.clock = pygame.time.Clock()
        self.running = False
    def main(self):
        while self.running:
            # fill the screen with a color to wipe away anything from last frame
            self.screen.fill(pygame.color.Color(255,255,255))

            #draw surfaces
            for thing in [self.noughts, self.crosses]: #this controls the layer order
                for position in thing[1]:
                    self.screen.blit(thing[0], position)
            
            # flip() the display to put your work on screen
            pygame.display.flip()
        exit()
    def update(self, model:Environment):
        self.noughts[1].clear()
        self.crosses[1].clear()
        for x in range(model.SIZE):
            for y in range(model.SIZE):
                match model.board[y][x]:
                    case Team.NOUGHT:
                        self.noughts[1].append((x*self.xSize, y*self.ySize))
                    case Team.CROSS:
                        self.crosses[1].append((x*self.xSize, y*self.ySize))