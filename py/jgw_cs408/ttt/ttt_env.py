#template from https://www.pygame.org/docs/
import pygame
from enum import Enum
from threading import Thread
from jgw_cs408.observer import Observable, Observer
import tensorflow as tf
import random

REWARD_TWO_IN_A_ROW = 50
REWARD_WIN = 100
REWARD_PER_TIME_STEP = 1
REWARD_INVALID = -100

#agent that follows the optimal policy by performing a full search
class SearchAgent():
    def  __init__(self, epsilon=0, epsilonDecay=0) -> None:
        self.epsilon=epsilon
        self.epsilonDecay = epsilonDecay
    def act(self, s) -> (0|1|2|3|4|5|6|7|8):
        #0.0 = Empty
        #1.0 = Player
        #2.0 = Enemy
        def actionSpace(s):
            actions = []
            for i in range(9):
                if s[0][i]==0:
                    actions.append(i)
            return actions
        def succ(s, a, actor):
            s = s.copy()
            s[0][a] = actor
            return s
        def isTerminal(s) -> (int|None):
            def win(s, actor):
                a = s[0][0]==actor
                b = s[0][1]==actor
                c = s[0][2]==actor
                d = s[0][3]==actor
                e = s[0][4]==actor
                f = s[0][5]==actor
                g = s[0][6]==actor
                h = s[0][7]==actor
                i = s[0][8]==actor
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
            def draw(s):
                for x in s[0]:
                    if x==0:
                        return False
                return True
            if win(s,1):
                return 999
            elif win(s,2):
                return -999
            elif draw(s):
                return 0
            else:
                return None
        def mini(s) -> int:
            tV = isTerminal(s)
            if tV==None: #not terminal
                worst = None
                for action in actionSpace(s):
                    score = max(succ(s,action, 2.0))
                    if score==-999:
                        return score
                    if worst==None or score<worst:
                        worst = score
                return worst
            else:
                return tV
        def max(s) -> int:
            tV = isTerminal(s)
            if tV==None:
                best = None
                for action in actionSpace(s):
                    score = mini(succ(s,action, 1.0))
                    if score==999:
                        return score
                    if best==None or score>best:
                        best = score
                return best
            else:
                return tV
        
        self.epsilon *= self.epsilonDecay
        if random.random()<self.epsilon: #chance to act randomly
            valid = []
            for i in range(9):
                if s[0][i] == 0.0:
                    valid.append(i)
            return random.choice(valid)
        else:
            s = s.numpy()
            best = None
            for action in actionSpace(s):
                score = mini(succ(s,action, 1.0))
                if best==None or score>best[1]:
                    best = (action, score)
            return None if best==None else best[0]

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
        def twoInARow(actor: Team) -> bool:
            a = self.board[0]==actor #T
            b = self.board[1]==actor #F
            c = self.board[2]==actor #T
            d = self.board[3]==actor #F
            e = self.board[4]==actor #F
            f = self.board[5]==actor #T
            g = self.board[6]==actor #F
            h = self.board[7]==actor #F
            i = self.board[8]==actor #F
            return (e and (a or b or c or d or f or g or h or i)) or (b and (a or c or d or f)) or (d and (a or g or h)) or (f and (c or h or i)) or (h and (g or i))
        def isWinning(actor : Team) -> bool:
            #a b c
            #d e f
            #g h i
            a = self.board[0]==actor
            b = self.board[1]==actor
            c = self.board[2]==actor
            d = self.board[3]==actor
            e = self.board[4]==actor
            f = self.board[5]==actor
            g = self.board[6]==actor
            h = self.board[7]==actor
            i = self.board[8]==actor
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
                    reward = REWARD_WIN #your winner!
                    self.truncated = True
                else: #end game as a draw
                    self.terminated = Team.EMPTY not in self.board
            else:
                validAction = False
                reward = REWARD_INVALID #don't make invalid moves
        self.notify() #redraw
        reward+=REWARD_PER_TIME_STEP #for time
        #for 2-in-a-row
        if twoInARow(actor):
            reward += REWARD_TWO_IN_A_ROW
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
        self.noughts[0] = pygame.transform.scale(pygame.image.load("jgw_cs408/img/nought.png"), (self.xSize+1, self.ySize+1)) #TODO transparency doesn't work, probably a file format thing

        #cross surface
        self.crosses[0] = pygame.transform.scale(pygame.image.load("jgw_cs408/img/cross.png"), (self.xSize+1, self.ySize+1)) #TODO transparency doesn't work, probably a file format thing

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
        self.actionSpace = range(9)
        self.model = TTTModel()
        if (render_mode=="human"):
            self.view = TTTView(resolution=(600,600))
            self.model.addObserver(self.view)
            self.view.open()
        else:
            self.view = None
        self.opponent = opponent
    def opponentAct(self, opponent=None):
        if opponent==None: #use own opponent if none was provided
            opponent=self.opponent
        #get a valid action from the opponent and return it
        opponentActionValid = False
        while not opponentActionValid:
            opponentAction = opponent.act(tf.expand_dims(tf.convert_to_tensor(self.model.calcLogits(Team.CROSS)),0))
            s2, _, terminated, truncated, opponentActionValid = self.model.step(Team.CROSS, opponentAction) #handle CPU action
        return (s2, _, terminated, truncated, None)
    def step(self, action):
        if not action == None:
            s1, rew, terminated, truncated, playerValid = self.model.step(Team.NOUGHT, action) #handle player action
            if playerValid and not (terminated or truncated):
                return self.opponentAct()
            else:
                self.model.terminated = True
                return (s1, rew, True, truncated, None) #return w/ invalid action warning
    def close(self):
        if not self.view == None:
            self.view.close()
            self.view = None