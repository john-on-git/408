#template from https://www.pygame.org/docs/
import pygame
from enum import Enum
from threading import Thread
from observer import Observable, Observer
from functools import reduce

#models
class Team(Enum):
    EMPTY = 0,
    NOUGHT = 1,
    CROSS = 2
class TTTModel (Observable):
    def __init__(self) -> None:
        super().__init__()
        self.board = [Team.EMPTY] * 9
        self.terminated = False
        self.truncated = False
    def reset(self) -> None:
        self.board = [Team.EMPTY] * 9
    def step(self, actor : Team, action : (0|1|2|3|4|5|6|7|8)) -> (tuple, int, bool, bool, None):
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
                    (a and (b and c) or (d and g)) or
                    (e and (d and f) or (b and h)) or 
                    (i and (h and g) or (c and f))
                ) or (
                    #diagonals
                    e and (
                        (a and i) or
                        (b and h) or
                        (d and f) or
                        (g and c)
                    )
                )
        def calcLogits(actor :Team) -> [float]:
            def logit(actor : Team, x:Team):
                if x==Team.EMPTY:
                    return 0.0 #empty
                elif x==actor:
                    return 1.0 #player
                else:
                    return 2.0 #enemy
            return [logit(actor, x) for x in self.board]

        reward = 0
        validAction = True
        if not self.terminated and not self.truncated:
            if self.board[action] == Team.EMPTY: #enact move if valid
                self.board[action] = actor
                if isWinning(actor): #end game if there's a winner
                    reward = 999 #your winner!
                    self.truncated = True
                else: #end game as a draw
                    self.terminated = reduce((lambda x,y: x==Team.NOUGHT or x==Team.CROSS or (x and not y==Team.EMPTY)), self.board) #end of game because all squares are field
                self.notify()
            else:
                validAction = False
                reward = -1 #don't make invalid moves
        return (calcLogits(actor), reward, self.terminated, self.truncated, validAction)
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
        for cell in range(len(model.board)):
            x = cell%3
            y = (cell-x)/3
            match model.board[cell]:
                case Team.NOUGHT:
                    self.noughts[1].append((x*self.xSize, y*self.ySize))
                case Team.CROSS:
                    self.crosses[1].append((x*self.xSize, y*self.ySize))

#agent that follows the optimal policy by performing a full search
class TTTSearchAgent():
    def act(self, s : [int]) -> (0|1|2|3|4|5|6|7|8):
        #0.0 = Empty
        #1.0 = Player
        #2.0 = Enemy
        def actionSpace(s):
            actions = []
            for i in range(len(s)):
                if s[i]==0:
                    actions.append(i)
            return actions
        def succ(s, a):
            s = s.copy()
            s[a] = 1.0
            return s
        def isTerminal(s : [int]) -> (int|None):
            def loss(s):
                a = s[0]==2
                b = s[1]==2
                c = s[2]==2
                d = s[3]==2
                e = s[4]==2
                f = s[5]==2
                g = s[6]==2
                h = s[7]==2
                i = s[8]==2
                return (
                        #cardinals
                        (a and (b and c) or (d and g)) or
                        (e and (d and f) or (b and h)) or 
                        (i and (h and g) or (c and f))
                    ) or (
                        #diagonals
                        e and (
                            (a and i) or
                            (b and h) or
                            (d and f) or
                            (g and c)
                        )
                    )
            def win(s):
                a = s[0]==1
                b = s[1]==1
                c = s[2]==1
                d = s[3]==1
                e = s[4]==1
                f = s[5]==1
                g = s[6]==1
                h = s[7]==1
                i = s[8]==1
                return (
                        #cardinals
                        (a and (b and c) or (d and g)) or
                        (e and (d and f) or (b and h)) or 
                        (i and (h and g) or (c and f))
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
                for x in s:
                    if x==0:
                        return False
                return True
            if loss(s):
                return -999
            elif win(s):
                return 999
            elif draw(s):
                return 0
            else:
                return None
        def mini(s : [int]) -> int:
            tV = isTerminal(s)
            if tV==None:
                worst = None
                for action in actionSpace(s):
                    score = max(succ(s,action))
                    if worst==None or score<worst:
                        worst = score
                return worst
            else:
                return tV
        def max(s : [int]) -> int:
            tV = isTerminal(s)
            if tV==None:
                best = None
                for action in actionSpace(s):
                    score = mini(succ(s,action))
                    if best==None or score>best:
                        best = score
                return best
            else:
                return tV
        best = None
        for action in actionSpace(s):
            score = mini(succ(s,action))
            if best==None or score>best[1]:
                best = (action, score)
        return best[0]
class TTTEnv():
    def reset(self) -> None:
        self.model.reset()
        return (self.model.calcLogits(), None)
    def __init__(self, render_mode : (None|str)=None) -> None:
        self.model = TTTModel()
        if (render_mode=="human"):
            self.view = TTTView(resolution=(600,600))
            self.model.addObserver(self.view)
            self.view.open()
        else:
            self.view = None
        self.opponent = TTTSearchAgent()
    def step(self, action):
        if not self.view == None: 
            # pygame.QUIT event means the user clicked X to close your window (this needs to be in the main thread for some reason)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
        s1, rew, terminated, truncated, valid = self.model.step(Team.NOUGHT, action) #handle player action
        if valid:
            s2, pun, terminated, truncated, valid = self.model.step(Team.CROSS, self.opponent.act(s1)) #handle CPU action
            return (s2, rew-pun, terminated, truncated, valid)
        else:
            return (s1, rew, terminated, truncated, valid)
    def close(self):
        if not self.view == None:
            self.view.close()
            self.view = None