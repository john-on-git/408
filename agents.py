import tensorflow as tf
from keras import Model
from keras import layers
import random
import tensorflow_probability as tfp
from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def act(self,s):
        pass
    @abstractmethod
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch, callbacks=[]):
        pass

#Non-reinforcement learning agents.
#this agent chooses actions at random w/ equal probability.
class RandomAgent(Agent):
    def __init__(self, validActions, **kwargs):
        self.epsilon = 1
        self.validActions = validActions
    def act(self, s):
        return random.choice(self.validActions(s))
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch, callbacks=[]):
        pass
    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        pass

class TTTSearchAgent(): #agent that follows the optimal policy by performing a tree search
    def  __init__(self, random:random.Random, epsilon=0, epsilonDecay=1, depth=-1) -> None:
        self.epsilon=epsilon
        self.epsilonDecay = epsilonDecay
        self.depth=depth
        self.random = random
    def act(self, s) -> int: #state is a tensor, so that the search agent has the same interface as a NN agent, and can be swapped out for one  
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
            for a in actionSpace(s):
                if s[0][a] == 0.0:
                    valid.append(a)
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

class OptimalMazeAgent(): #optimal policy for environments with coins. This ignores the exploration reward so it's not strictly optimal, but it's very close.
    def __init__(self) -> None:
        super().__init__()
    def act(self,s):
        def h(node): #A* heuristic: manhattan distance between agent and coin
            return abs(node[0]-coinCoords[0]) + abs(node[1]-coinCoords[1])
        def neighbours(node):
            y,x = node
            neighbours = []
            for action,nextCoords in [(0, (y-1,x)),(1, (y,x-1)),(2, (y+1,x)),(3, (y,x+1))]: #up, left, down, right
                #if it's possible to move to here
                if (nextCoords[0]>=0) and (nextCoords[0]<len(s)) and (nextCoords[1]>=0) and (nextCoords[1]<len(s)) and (s[nextCoords[0]][nextCoords[1]][LOGIT_SOLID]==0):
                    neighbours.append((action, nextCoords))
            return neighbours
        LOGIT_PLAYER = 0
        LOGIT_COIN = 2
        LOGIT_SOLID = 3
        agentCoords = None
        y = 0
        while agentCoords == None and y < (len(s)): #get our coords
            x = 0
            while agentCoords == None and x < (len(s[y])):
                if s[y][x][LOGIT_PLAYER]==1:
                    agentCoords = (y,x)
                x+=1
            y +=1
        lowestDist = len(s)*2 + 1
        coinCoords = None
        for y in range(len(s)):
            for x in range(len(s[y])):
                if s[y][x][LOGIT_COIN]==1: #if there's a coin here
                    dist = abs(y-agentCoords[0]) + abs(x-agentCoords[1])
                    if dist<lowestDist:
                        coinCoords = (y,x)
                        lowestDist=dist
        assert coinCoords!=None
        #A* search
        frontier = set()
        frontier.add(agentCoords)
        prev = {} #(action required to go from value to key, (x,y))
        f = {}
        g = {agentCoords: 0}
        while len(frontier)>0:
            #get the cheapest node from frontier
            node = None
            for other in frontier:
                if node==None or g[node]>g[other]:
                    node = other
            frontier.remove(node)
            for action, neighbour in neighbours(node): #check all neighbours
                gNeighbour = g[node] + 1
                if neighbour not in g or gNeighbour<g[neighbour]: #if new best path found
                    prev[neighbour] = (action, node)
                    g[neighbour] = gNeighbour
                    f[neighbour] = gNeighbour + h(neighbour)
                    if neighbour not in frontier: #add new node if it's not already in frontier
                        frontier.add(neighbour)
        backtrack = (4, coinCoords) #re-trace steps
        while backtrack[1] in prev:
            backtrack = prev[backtrack[1]]
        return backtrack[0]

#Reinforcement learning agents.
class AbstractQAgent(Model, Agent):
    def __init__(self, learningRate, actionSpace, hiddenLayers, validActions, epsilon, epsilonDecay, discountRate):
        super().__init__()
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.discountRate = discountRate
        self.actionSpace = actionSpace
        self.validActions = validActions if validActions is not None else (lambda _: actionSpace) #Callable that returns the valid actions for a state. Defaults to entire action space. That lambda can't be a default value because the definition references actionSpace.

        #initLayers
        self.modelLayers = []
        self.modelLayers.extend(hiddenLayers)
        self.modelLayers.append(layers.Dense(len(actionSpace)))
        self.compile(
            optimizer=tf.optimizers.Adam(learning_rate=learningRate),
            metrics="loss"
        )
    def call(self, s):
        for layer in self.layers:
            s = layer(s)
        return s
    def act(self, s):
        validActions = self.validActions(s) #get valid actions
        if random.random()<self.epsilon: #chance to act randomly
            return random.choice(validActions)
        else:
            #manually reject actions that are not valid
            vals = self(s)[0]
            newVals = [min(vals)-1] * len(vals)
            for i in validActions:
                newVals[i] = vals[i]
            return int(tf.argmax(newVals)) #follow greedy policy
    @abstractmethod
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch, callbacks=[]):
        pass
class AbstractPolicyAgent(Model, Agent):
    def __init__(self, learningRate, actionSpace, hiddenLayers, validActions, epsilon, epsilonDecay, discountRate):
        super().__init__()
        self.discountRate = discountRate
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.actionSpace = actionSpace
        self.validActions = validActions if validActions is not None else (lambda _: actionSpace) #Callable that returns the valid actions for a state. Defaults to entire action space. That lambda can't be a default value because the definition references actionSpace.

        #initLayers
        self.modelLayers = []
        self.modelLayers.extend(hiddenLayers)
        self.modelLayers.append(layers.Dense(len(actionSpace), activation=tf.nn.softmax))
        self.compile(
            optimizer=tf.optimizers.Adam(learning_rate=learningRate),
            metrics="loss"
        )
    def call(self, s):
        for layer in self.layers:
            s = layer(s)
        return s
    def act(self, s):
        validActions = self.validActions(s) #get valid actions
        if random.random()<self.epsilon: #chance to act randomly
            return random.choice(validActions)
        else:
            #manually reject actions that are not valid
            probs = self(s)[0]
            newProbs = [0.0] * len(probs)
            for i in validActions:
                newProbs[i] = probs[i]
            return tfp.distributions.Categorical(probs=newProbs).sample() #follow greedy policy
    @abstractmethod
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch, callbacks=[]):
        pass
class AbstractActorCriticAgent(Model, Agent):
    def __init__(self, learningRate, actionSpace, validActions, epsilon, epsilonDecay, discountRate, criticWeight, entropyWeight):
        super().__init__()
        self.discountRate = discountRate
        self.learningRate = learningRate
        self.actionSpace = actionSpace
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.criticWeight = criticWeight
        self.entropyWeight = entropyWeight
        self.validActions = validActions if validActions is not None else (lambda _: actionSpace) #Callable that returns the valid actions for a state. Defaults to entire action space.

        #layers are initialised in subclasses
    def call(self, s):
        for layer in self.layers:
            s = layer(s)
        return s
    def act(self, s):
        validActions = self.validActions(s) #get valid actions
        if random.random()<self.epsilon: #chance to act randomly
            return random.choice(validActions)
        else:
            probs = tf.nn.softmax(self(s)[0][:len(self.actionSpace)]) #ignore Q-vals and take probs
            newProbs = [0.0] * len(probs)
            for i in validActions:
                newProbs[i] = probs[i]
            return tfp.distributions.Categorical(probs=newProbs).sample()

#From Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning, Williams, 1992.
class REINFORCEAgent(AbstractPolicyAgent):
    def __init__(self, learningRate, actionSpace, hiddenLayers, validActions=None, epsilon=0, epsilonDecay=1, discountRate=1, baseline=0.0, entropyWeight=1, **kwargs):
        super().__init__(learningRate, actionSpace, hiddenLayers, validActions, epsilon, epsilonDecay, discountRate)
        self.baseline = baseline
        self.entropyWeight = entropyWeight
    def train_step(self, data):
        def l():
            s,a,r,h = data
            return -(r - self.baseline + self.entropyWeight*h) * tf.math.log(self(s)[0][a]) #negate, because this is a loss & we are trying to minimize it
        self.optimizer.minimize(l, self.trainable_weights)
        return {"loss": l()}
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch, callbacks=[]):
        def discountedFutureRewards(rs):
            discRs = [None] * len(rs)
            r = 0
            for i in range(len(rs)-1,-1,-1):
                r*=self.discountRate
                r+=rs[i]
                discRs[i] = r
            return discRs
        def entropies(ss):
            entropies = []
            for s in ss: #sum up the entropy of each state
                entropy = 0
                p = self(s)[0]
                for a in range(len(self.actionSpace)):
                    entropy-= p[a] * tf.math.log(p[a])
                entropies.append(entropy)
            return entropies
        #epoch ends, reset env, observation, & reward
        if endOfEpoch:
            self.epsilon *= self.epsilonDecay #epsilon decay
            #train model
            dataset = tf.data.Dataset.from_tensor_slices((
                observationsThisEpoch[:-1],
                actionsThisEpoch,
                discountedFutureRewards(rewardsThisEpoch),
                entropies(observationsThisEpoch[:-1])
            ))
            self.fit(dataset) #train on the minibatch

#value-based
#From Playing Atari with Deep Reinforcement Learning, Mnih et al (Algorithm 1).
class DQNAgent(AbstractQAgent):
    def __init__(self, learningRate, actionSpace, hiddenLayers, validActions=None, epsilon=0, epsilonDecay=1, discountRate=1, replayMemoryCapacity=1000, replayFraction=5, **kwargs):
        super().__init__(learningRate, actionSpace, hiddenLayers, validActions, epsilon, epsilonDecay, discountRate)
        self.replayMemoryS1s = []
        self.replayMemoryA1s = []   
        self.replayMemoryRs  = []
        self.replayMemoryS2s = []
        self.replayMemoryCapacity = replayMemoryCapacity
        self.replayFraction = replayFraction
        self.flag = True
    def train_step(self, data):
        def l(): #from atari paper
            s1,a1,r,s2 = data
            q2 = self.discountRate*tf.reduce_max(self(s2)[0]) #estimated q-value for on-policy action for s2
            q1 = self(s1)[0][a1] #estimated q-value for (s,a) yielding r
            return (r+q2-q1)**2
        self.optimizer.minimize(l, self.trainable_weights)
        return {"loss": l()}
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch, callbacks=[]):
        if len(observationsThisEpoch)>1: #if we have a transition to add
            #removing this causes a crash (sometimes?).
            #Probably something to do with the layer initialisation, but .fit just calls this? No clue and no time to fix it.
            #Might indicate that the whole thing is structured wrong, idk. 🤷‍♂️
            if self.flag:
                self.flag = False
                self.train_step((observationsThisEpoch[-2], actionsThisEpoch[-1], rewardsThisEpoch[-1], observationsThisEpoch[-1]))
            #add the transition
            self.replayMemoryS1s.append(observationsThisEpoch[-2])
            self.replayMemoryA1s.append(actionsThisEpoch[-1])
            self.replayMemoryRs.append(rewardsThisEpoch[-1])
            self.replayMemoryS2s.append(observationsThisEpoch[-1])
            if len(self.replayMemoryS1s)>self.replayMemoryCapacity: #if this puts us over capacity remove the oldest transition to put us back under cap
                self.replayMemoryS1s.pop(0)
                self.replayMemoryA1s.pop(0)
                self.replayMemoryRs.pop(0)
                self.replayMemoryS2s.pop(0)
            
            if endOfEpoch:
                self.epsilon *= self.epsilonDecay #epsilon decay
                #build the minibatch
                miniBatchS1s = []
                miniBatchAs  = []
                miniBatchRs  = []
                miniBatchS2s = []
                sample = random.sample(range(len(self.replayMemoryS1s)), min(len(self.replayMemoryS1s), int(self.replayMemoryCapacity/self.replayFraction)))
                for i in sample:
                    miniBatchS1s.append(self.replayMemoryS1s[i])
                    miniBatchAs.append(self.replayMemoryA1s[i])
                    miniBatchRs.append(self.replayMemoryRs[i])
                    miniBatchS2s.append(self.replayMemoryS2s[i])
                dataset = tf.data.Dataset.from_tensor_slices((miniBatchS1s, miniBatchAs, miniBatchRs, miniBatchS2s))
                self.fit(dataset, batch_size=int(self.replayMemoryCapacity/self.replayFraction), callbacks=callbacks) #train on the minibatch
#Same as above but adapted based on SARSA coverage in Sutton/Barto. Like DQN, but learns on-policy.
class SARSAAgent(AbstractQAgent):
    def __init__(self, learningRate, actionSpace, hiddenLayers, validActions=None, epsilon=0, epsilonDecay=1, discountRate=1, **kwargs):
        super().__init__(learningRate, actionSpace, hiddenLayers, validActions, epsilon, epsilonDecay, discountRate)
    def train_step(self, data):
        def l(): #from atari paper
            s1,a1,r,s2,a2 = data
            q2 = self.discountRate*self(s2)[0][a2] #estimated q-value for on-policy action for s2
            q1 = self(s1)[0][a1] #estimated q-value for (s,a) yielding r
            return tf.math.squared_difference(r+q2, q1)
        self.optimizer.minimize(l, self.trainable_weights)
        return {"loss": l()}
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch, callbacks=[]):
        if endOfEpoch:
            self.epsilon *= self.epsilonDecay #epsilon decay
        if len(observationsThisEpoch)>2: #if we have a transition to add
            self.train_step((observationsThisEpoch[-3], actionsThisEpoch[-2], rewardsThisEpoch[-2], observationsThisEpoch[-2],actionsThisEpoch[-1]))

#From Actor-Critic Algorithms, Konda & Tsitsiklis, 1999.
class ActorCriticAgent(AbstractActorCriticAgent):
    def __init__(self, learningRate, actionSpace, hiddenLayers, validActions=None, epsilon=0, epsilonDecay=1, discountRate=1, replayMemoryCapacity=1000, replayFraction=5, entropyWeight=1, criticWeight=1, **kwargs):
        super().__init__(learningRate, actionSpace, validActions, epsilon, epsilonDecay, discountRate, criticWeight, entropyWeight)
        self.replayMemoryS1s = []
        self.replayMemoryA1s = []
        self.replayMemoryRs  = []
        self.replayMemoryS2s = []
        self.replayMemoryA2s = []
        self.replayMemoryCapacity = replayMemoryCapacity
        self.replayFraction = replayFraction
        #init layers
        self.modelLayers = []
        self.modelLayers.extend(hiddenLayers)
        self.modelLayers.append(layers.Dense(len(actionSpace)*2)) #first half is interpreted as action probs, second half as action Q-values. Softmax activation is manually applied, and only to probs.
        self.compile(
            optimizer=tf.optimizers.Adam(learning_rate=learningRate),
            metrics="loss"
        )
    @tf.function
    def train_step(self, data):
        def l():
            s1,a,r,s2,h = data #unpack transition

            #calculate critic loss (T-D MSE)
            q2 = self.discountRate*tf.reduce_max(self(s2)[0][len(self.actionSpace):]) #estimated Q-value for on-policy action for s2
            q1 = self(s1)[0][len(self.actionSpace):][a] #estimated Q(s,a)
            lC = (r + self.discountRate*q2 - q1)**2

            #calculate actor loss
            lA =  (r + self.discountRate*q2) * tf.math.log(tf.nn.softmax(self(s1)[0][:len(self.actionSpace)])[a]) #apply gradients inverts the gradient, so it must be inverted here as well
            return -(lA - self.criticWeight*lC + self.entropyWeight*h)
        
        self.optimizer.minimize(l, self.trainable_weights)
        return {"loss": l()} #TODO
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch, callbacks=[]): 
        def entropies(Ss):
            entropies = []
            for s in Ss: #sum up the entropy of each state
                entropy = 0
                p = tf.nn.softmax(self(s)[0][:len(self.actionSpace)])
                for a in range(len(self.actionSpace)):
                    entropy-= p[a] * tf.math.log(p[a])
                entropies.append(entropy)
            return entropies
        if len(observationsThisEpoch)>1: #if we have a transition to add
            #add the transition
            self.replayMemoryS1s.append(observationsThisEpoch[-2])
            self.replayMemoryA1s.append(actionsThisEpoch[-1])
            self.replayMemoryRs.append(rewardsThisEpoch[-1])
            self.replayMemoryS2s.append(observationsThisEpoch[-1])
            if len(self.replayMemoryS1s)>self.replayMemoryCapacity: #if this puts us over capacity remove the oldest transition to put us back under cap
                self.replayMemoryS1s.pop(0)
                self.replayMemoryA1s.pop(0)
                self.replayMemoryRs.pop(0)
                self.replayMemoryS2s.pop(0)
            
            if endOfEpoch:
                self.epsilon *= self.epsilonDecay #epsilon decay
                #TODO self.train_step((observationsThisEpoch[-2], actionsThisEpoch[-2], rewardsThisEpoch[-2], observationsThisEpoch[-1])) #load-bearing train_step
                #build the minibatch
                miniBatchS1s = []
                miniBatchAs  = []
                miniBatchRs  = []
                miniBatchS2s = []
                
                for i in random.sample(range(len(self.replayMemoryS1s)), min(len(self.replayMemoryS1s), int(self.replayMemoryCapacity/self.replayFraction))):
                    miniBatchS1s.append(self.replayMemoryS1s[i])
                    miniBatchAs.append(self.replayMemoryA1s[i])
                    miniBatchRs.append(self.replayMemoryRs[i])
                    miniBatchS2s.append(self.replayMemoryS2s[i])
                miniBatchHs = entropies(miniBatchS1s)
                dataset = tf.data.Dataset.from_tensor_slices((miniBatchS1s, miniBatchAs, miniBatchRs, miniBatchS2s, miniBatchHs))
                self.fit(dataset) #train on the minibatch
#Synchronous verson of A3C, From Asynchronous Methods for Deep Reinforcement Learning.
class AdvantageActorCriticAgent(AbstractActorCriticAgent):
    def __init__(self, learningRate, actionSpace, hiddenLayers, validActions=None, epsilon=0, epsilonDecay=1, discountRate=1, entropyWeight=1, criticWeight=1, tMax=1000, **kwargs):
        super().__init__(learningRate, actionSpace, validActions, epsilon, epsilonDecay, discountRate, criticWeight, entropyWeight)
        self.tMax = tMax
        #init layers
        self.modelLayers = []
        self.modelLayers.extend(hiddenLayers)
        self.modelLayers.append(layers.Dense(len(actionSpace)+1)) #last output is the state V-value, all others are the action probs 
        self.compile(
            optimizer=tf.optimizers.Adam(learning_rate=self.learningRate),
            metrics="loss"
        )
    def train_step(self, data):
        def l():
            s,a,r,h = data
            p = tf.nn.softmax(self(s)[0][:-1])[a]
            adv = r - self(s)[0][-1]
            lA = (adv + self.entropyWeight*h) * tf.math.log(p) #characteristic eligibility. apply_gradients inverts the gradient, so it must be inverted here as well
            lC = adv**2
            return -(lA - self.criticWeight*lC)
        self.optimizer.minimize(l, self.trainable_weights)
        return  {"loss": l()} #TODO
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch, callbacks=[]):
        def entropies(Ss):
            entropies = []
            for s in Ss: #sum up the entropy of each state
                entropy = 0
                p = tf.nn.softmax(self(s)[0][:-1])
                for a in range(len(self.actionSpace)):
                    entropy-= p[a] * tf.math.log(p[a])
                entropies.append(entropy)
            return entropies
        def discountedFutureRewards(rs):
            discRs = [None] * len(rs)
            r = 0 if endOfEpoch else self(observationsThisEpoch[-1])[0][-1] #bootstrap based on estimated value of most recent state. If it's terminal we know the value is 0.
            for i in range(len(rs)-1,-1,-1):
                r*=self.discountRate
                r+=rs[i]
                discRs[i] = r
            return discRs
        if (len(observationsThisEpoch)%self.tMax==0) or endOfEpoch:
            m = len(observationsThisEpoch)%(self.tMax)
            trajectoryLength = self.tMax if m==0 else m

            ssSlice = observationsThisEpoch[-trajectoryLength-1:-1]
            asSlice = actionsThisEpoch[-trajectoryLength:]
            rsSlice = discountedFutureRewards(rewardsThisEpoch[-trajectoryLength:])
            hs = entropies(ssSlice)

            dataset = tf.data.Dataset.from_tensor_slices((
                ssSlice,
                asSlice,
                rsSlice,
                hs
            ))
            self.fit(dataset) #train on the minibatch
        if endOfEpoch:
            self.epsilon *= self.epsilonDecay #epsilon decay
#From Proximal Policy Optimization Algorithms,  TODO
class PPOAgent(AbstractActorCriticAgent):
    def __init__(self, learningRate, actionSpace, hiddenLayers, validActions, epsilon, epsilonDecay, discountRate, entropyWeight=1, criticWeight=2, tMax=1000, interval=0.2, **kwargs):
        super().__init__(learningRate, actionSpace, validActions, epsilon, epsilonDecay, discountRate, criticWeight, entropyWeight)
        self.tMax = tMax
        self.interval = interval
        self.actorLayers = []
        self.actorLayers.extend(hiddenLayers)
        self.actorLayers.append(layers.Dense(len(actionSpace)+1)) #last output is the state V-value, all others are the action probs
        self.compile(
            optimizer=tf.optimizers.Adam(learning_rate=self.learningRate),
            metrics="loss"
        )
    def train_step(self, data):
        def l():
            s,a,r,h,pOld = data
            adv = r - self(s)[0][-1]
            lVF = adv**2
            #calc rt(θ)
            pNew = tf.nn.softmax(self(s)[0][:-1])[a]
            rt = pNew/pOld
            lCLIP = tf.minimum(rt * adv, tf.clip_by_value(rt, 1-self.interval, 1+self.interval) * adv)
            s = self.entropyWeight*h * tf.math.log(pNew)
            return -(lCLIP - self.criticWeight*lVF + s) #lower bound, clip and stuff. This is the objective function from the paper, so it must be inverted to make it a loss function.
        self.optimizer.minimize(l, self.trainable_weights)
        return {"loss": l()}
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch, callbacks=[]):
        def entropies(Ss):
            entropies = []
            for s in Ss: #sum up the entropy of each state
                entropy = 0
                p = tf.nn.softmax(self(s)[0][:-1])
                for a in range(len(self.actionSpace)):
                    entropy-= p[a] * tf.math.log(p[a])
                entropies.append(entropy)
            return entropies
        def discountedFutureRewards(rs):
            discRs = [None] * len(rs)
            r = 0 if endOfEpoch else self(observationsThisEpoch[-1])[0][-1] #bootstrap based on estimated value of most recent state. If it's terminal we know the value is 0.
            for i in range(len(rs)-1,-1,-1):
                r*=self.discountRate
                r+=rs[i]
                discRs[i] = r
            return discRs
        if (len(observationsThisEpoch)%self.tMax==0) or endOfEpoch:
            m = len(observationsThisEpoch)%(self.tMax)
            trajectoryLength = self.tMax if m==0 else m

            s1sSlice = observationsThisEpoch[-trajectoryLength-1:-1]
            actionsSlice = actionsThisEpoch[-trajectoryLength:]
            rsSlice = discountedFutureRewards(rewardsThisEpoch[-trajectoryLength:])
            hs = entropies(s1sSlice)
            pOlds = [tf.nn.softmax(self(s)[0][:-1])[a] for s,a in zip(s1sSlice, actionsSlice)]

            dataset = tf.data.Dataset.from_tensor_slices((
                s1sSlice,
                actionsSlice,
                rsSlice,
                hs,
                pOlds
            ))
            self.fit(dataset) #train on the minibatch
        if endOfEpoch:
            self.epsilon *= self.epsilonDecay #epsilon decay