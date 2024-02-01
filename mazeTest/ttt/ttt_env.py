#template from https://www.pygame.org/docs/
import pygame
from enum import Enum
from threading import Thread
from observer import Observable, Observer
from functools import reduce
import tensorflow as tf
from ttt_agents import SearchAgent

#models
class Team(Enum):
    EMPTY = 0.0,
    NOUGHT = 1.0,
    CROSS = 2.0
class TTTModel (Observable):
    def __init__(self) -> None:
        super().__init__()
        self.board = [Team.EMPTY] * 9
        self.terminated = False
        self.truncated = False
    def reset(self) -> None:
        self.board = [Team.EMPTY] * 9
        self.truncated = False
        self.terminated = False
    def calcLogits(self, actor :Team) -> [float]:
        def logit(actor : Team, x:Team):
            if x==Team.EMPTY:
                return 0.0 #empty
            elif x==actor:
                return 1.0 #player
            else:
                return 2.0 #enemy
        return [logit(actor, x) for x in self.board]

    def step(self, actor : Team, action : (0|1|2|3|4|5|6|7|8)) -> (tuple, int, bool, bool, None):
        def isWinning(actor : Team) -> bool:
            #a b c
            #d e f
            #g h i
            a = self.board[0]==actor #T
            b = self.board[1]==actor #F
            c = self.board[2]==actor #T
            d = self.board[3]==actor #F
            e = self.board[4]==actor #F
            f = self.board[5]==actor #T
            g = self.board[6]==actor #F
            h = self.board[7]==actor #F
            i = self.board[8]==actor #F
            return (
                    #cardinals
                    (a and ((b and c) or (d and g))) or
                    (e and ((d and f) or (b and h))) or 
                    (i and ((h and g) or (c and f)))
                ) or (
                    #diagonals
                    e and (
                        (a and i) or
                        (b and h) or
                        (d and f) or
                        (g and c)
                    )
                )
        reward = 0
        validAction = True
        if not self.terminated and not self.truncated:
            if self.board[action] == Team.EMPTY: #enact move if valid
                self.board[action] = actor
                if isWinning(actor): #end game if there's a winner
                    reward = 1000 #your winner!
                    self.truncated = True
                else: #end game as a draw
                    self.terminated = reduce((lambda x,y: x==Team.NOUGHT or x==Team.CROSS or (x and not y==Team.EMPTY)), self.board) #end of game because all squares are field
            else:
                validAction = False
                reward = -10 #don't make invalid moves
        self.notify() #redraw
        reward-=1 #for time
        return (self.calcLogits(actor), reward, self.terminated, self.truncated, validAction)
    def notify(self) -> None:
        observer: Observer
        for observer in super().getObservers():
            observer.update(self)
#view
class TTTView(Observer):
    def __init__(self, resolution) -> None:
        super().__init__()
        pygame.init()
        displayResolution = resolution
        self.screen = pygame.display.set_mode(displayResolution)
        self.xSize = displayResolution[0]/3 #horizontal size of board
        self.ySize = displayResolution[1]/3 #vertical size of board
        self.noughts = [None, []]
        self.crosses = [None, []]

        #nought surface
        self.noughts[0] = pygame.Surface((self.xSize+1,self.ySize+1))
        noughtImage = pygame.transform.scale(pygame.image.load("img/nought.png"), (self.xSize, self.ySize)) #TODO transparency doesn't work, probably a file format thing
        noughtImage.set_colorkey((0, 0, 0))
        noughtImage = pygame.Surface.convert_alpha(noughtImage)
        self.noughts[0].blit(noughtImage, (0,0))

        #cross surface
        self.crosses[0] = pygame.Surface((self.xSize, self.ySize))
        crossImage = pygame.transform.scale(pygame.image.load("img/cross.png"), (self.xSize, self.ySize)) #TODO transparency doesn't work, probably a file format thing
        crossImage.set_colorkey((0, 0, 0))
        crossImage = pygame.Surface.convert_alpha(crossImage)
        self.crosses[0].blit(crossImage, (0,0))

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
    def open(self):
        self.running = True
        self.thread = Thread(target=self.main, daemon=True)
        self.thread.start()
    def close(self):
        self.running = False
        self.thread.join()
        pygame.quit()
    def update(self, model):
        self.noughts[1].clear()
        self.crosses[1].clear()
        for cell in range(len(model.board)):
            x = cell%3
            y = (cell-x)/3
            match model.board[cell]:
                case Team.NOUGHT:
                    self.noughts[1].append((x*self.xSize, y*self.ySize))
                case Team.CROSS:
                    self.crosses[1].append((x*self.xSize, y*self.ySize))

class TTTEnv():
    def reset(self, seed=None) -> None:
        self.model.reset()
        return (self.model.calcLogits(Team.NOUGHT), None)
    def __init__(self, render_mode : (None|str)=None, opponent=SearchAgent()) -> None:
        self.model = TTTModel()
        if (render_mode=="human"):
            self.view = TTTView(resolution=(600,600))
            self.model.addObserver(self.view)
            self.view.open()
        else:
            self.view = None
        self.opponent = opponent
    def step(self, action):
        if not self.view == None: 
            # pygame.QUIT event means the user clicked X to close your window (this needs to be in the main thread for some reason)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
        if not action == None:
            s1, rew, terminated, truncated, playerValid = self.model.step(Team.NOUGHT, action) #handle player action
            if playerValid and not (terminated or truncated):
                #if the player's action was valid, get a valid action from the opponent and return it
                opponentActionValid = False
                while not opponentActionValid:
                    opponentAction = self.opponent.act(tf.expand_dims(tf.convert_to_tensor(self.model.calcLogits(Team.CROSS)),0))
                    s2, _, terminated, truncated, opponentActionValid = self.model.step(Team.CROSS, opponentAction) #handle CPU action
                return (s2, rew, terminated, truncated, True)
            else:
                return (s1, rew, terminated, truncated, False) #return w/ invalid action warning
    def close(self):
        if not self.view == None:
            self.view.close()
            self.view = None