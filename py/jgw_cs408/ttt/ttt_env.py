#template from https://www.pygame.org/docs/
from math import sqrt
import pygame
from enum import Enum
from threading import Thread
from jgw_cs408.observer import Observable, Observer
import tensorflow as tf
import random

REWARD_PARTIAL_CHAIN = 2
REWARD_WIN = 10
REWARD_PER_TIME_STEP = 1
REWARD_INVALID = -100

#agent that follows the optimal policy by performing a full search
class TTTSearchAgent():
    def  __init__(self, epsilon=0, epsilonDecay=1, depth=-1) -> None:
        self.epsilon=epsilon
        self.epsilonDecay = epsilonDecay
        self.depth=depth
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
            n = int(sqrt(len(s[0])))
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
                    return minScore
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
                    return maxScore
                else: #stop if terminal
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
            maxScore = (None, -float('inf'))
            for a in actionSpace(s):
                score = mini(succ(s,a,1.0), self.depth-1, -float('inf'), float('inf')) 
                if score==999:
                    return a
                elif score>maxScore[1]:
                    maxScore = (a, score)
            return maxScore[0]

#models
class Team(Enum):
    EMPTY = 0.0,
    NOUGHT = 1.0,
    CROSS = 2.0
class TTTModel (Observable):
    def __init__(self, size, opponent) -> None:
        super().__init__()
        self.size = size
        self.opponent = opponent
        self.board = []
        for i in range(self.size):
            self.board.append([])
            for _ in range(self.size):
                self.board[i].append(Team.EMPTY)
        self.terminated = False
        self.truncated = False
    def reset(self, _) -> None:
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                self.board[i][j] = Team.EMPTY
        self.truncated = False
        self.terminated = False
        self.notify() #update view
        return (self.calcLogits(Team.NOUGHT), None)
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
    def step(self, action): #both sides move
        if not action == None:
            s, rew, terminated, truncated, info = self.halfStep(Team.NOUGHT, action) #handle player action
            if not (terminated or truncated):
                opponentAction = self.opponent.act(tf.expand_dims(tf.convert_to_tensor(self.calcLogits(Team.CROSS)),0))
                if self.board[int(opponentAction/self.size)][opponentAction%self.size] == Team.EMPTY: #enact move if valid
                    s, _, terminated, truncated, info = self.halfStep(Team.CROSS, opponentAction) #handle CPU action
                    return (s, rew, terminated, truncated, info)
                else:
                    self.terminated = True
                    raise Exception("Tic-Tac-Toe opponent made invalid move.")
            else:
                return (s, rew, True, truncated, info) #last move of the game
    def halfStep(self, actor : Team, action : int) -> list[list[float], int, bool, bool, None]: #one side moves
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
        actX = action%self.size
        actY = int(action/self.size)
        if not self.terminated and not self.truncated:
            if self.board[actY][actX] == Team.EMPTY: #enact the move if it's valid
                self.board[actY][actX] = actor
                longestChains = calcLongestChains(self.size)
                if longestChains[actor]==self.size: #end the game if there's a winner
                    reward = REWARD_WIN**self.size #distribute reward for the win
                    self.truncated = True
                else:
                    #end the game as a draw if all cells are full
                    self.terminated = True
                    for row in self.board:
                        for cell in row:
                            if cell == Team.EMPTY:
                                self.terminated = False
                    reward += REWARD_PARTIAL_CHAIN**longestChains[actor] #distribute reward for partial chains
            else:
                reward = REWARD_INVALID #distribute negative reward for invalid moves
            reward+=REWARD_PER_TIME_STEP #distribute reward for time
        logits = self.calcLogits(actor)
        info = {}
        self.notify() #update view
        return (logits, reward, self.terminated, self.truncated, info)
    def notify(self) -> None:
        observer: Observer
        for observer in super().getObservers():
            observer.update(self)
#view
class TTTView(Observer):
    def __init__(self, resolution, model : TTTModel) -> None:
        super().__init__()
        pygame.init()
        model.addObserver(self)
        displayResolution = resolution
        self.screen = pygame.display.set_mode(displayResolution)
        self.xSize = displayResolution[0]/model.size #horizontal size of board
        self.ySize = displayResolution[1]/model.size #vertical size of board
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
        for x in range(model.size):
            for y in range(model.size):
                match model.board[y][x]:
                    case Team.NOUGHT:
                        self.noughts[1].append((x*self.xSize, y*self.ySize))
                    case Team.CROSS:
                        self.crosses[1].append((x*self.xSize, y*self.ySize))

class TTTEnv():
    def reset(self, seed=None) -> None:
        return self.model.reset(seed)
    def __init__(self, render_mode:(None|str)=None, size=3, opponent=TTTSearchAgent()) -> None:
        """
        Initialize a new TTTEnv.

        Args.
        render_mode: The environment will launch with a human-readable GUI if render_mode is exactly "human".
        size: Equal to the width and height of the game board.
        opponent: Agent responsible for the enemy AI. The provided agent be guaranteed to provide a valid action for all game states.
        """
        self.model = TTTModel(size, opponent)
        if (render_mode=="human"):
            self.view = TTTView(resolution=(600,600), model=self.model)
            self.view.open()
        else:
            self.view = None
    def step(self, action):
        self.model.step(action)
    def close(self):
        if not self.view == None:
            self.view.close()
            self.view = None
    def validActions(self,s):
        valid = []
        for action in range(self.model.size**2):
            if s[action] == 0.0:
                valid.append()
        return valid