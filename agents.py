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
#Replay method from Playing Atari with Deep Reinforcement Learning, Mnih et al (Algorithm 1).
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
            q2 = self.discountRate*tf.reduce_max(self(s2)) #estimated q-value for on-policy action for s2
            q1 = self(s1)[0][a1] #estimated q-value for (s,a) yielding r
            return tf.math.squared_difference(r+q2, q1)
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
#This is similar to DQN, but learns on-policy.
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

#From Actor-Critic Algorithms, Konda & Tsitsiklis, NIPS 1999.
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
#from Proximal Policy Optimisation (???) TODO
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