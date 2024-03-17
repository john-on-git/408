import tensorflow as tf
from keras import Model
from keras import layers
import random
import tensorflow_probability as tfp
import math
from abc import ABC, abstractmethod

#TODO
    #all agents need to implement blocking of invalid actions. This also needs to be applied to the training logic, i.e. called by q()

class Agent(ABC):
    @abstractmethod
    def act(self,s):
        pass

#this agent chooses actions at random w/ equal probability.
class RandomAgent(Agent):
    def __init__(self, actionSpace):
        self.epsilon = 1
        self.actionSpace = actionSpace
    def act(self, _):
        return random.choice(self.actionSpace)
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch, callbacks=[]):
        pass

class AbstractQAgent(Model, Agent):
    def __init__(self, learningRate, actionSpace, hiddenLayers, validActions, epsilon, epsilonDecay):
        super().__init__()
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
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
    def __init__(self, learningRate, actionSpace, hiddenLayers, validActions, epsilon, epsilonDecay):
        super().__init__()
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
    def __init__(self, learningRate, actionSpace, validActions, epsilon, epsilonDecay, discountRate, entropyWeight):
        super().__init__()
        self.discountRate = discountRate
        self.learningRate = learningRate
        self.actionSpace = actionSpace
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
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

#policy gradient methods
#from Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning, Williams, 1992.
#TODO if there's time, rework this to use a loss function (currently it doesn't due to my poor understanding of the topic at the time of implementation)
class REINFORCEAgent(AbstractPolicyAgent):
    def __init__(self, learningRate, actionSpace, hiddenLayers, validActions=None, epsilon=0, epsilonDecay=1, discountRate=1, baseline=0):
        super().__init__(learningRate, actionSpace, hiddenLayers, validActions, epsilon, epsilonDecay)
        self.discountRate = discountRate
        self.baseline = baseline
    def train_step(self, eligibilityTraces, r):
        #calculate sum of eligibility traces
        grads = [None] * len(eligibilityTraces[0])
        for eligibilityTrace in eligibilityTraces:
            for i in range(len(eligibilityTrace)):
                if grads[i] is None:
                    grads[i] = eligibilityTrace[i]
                else:
                    grads[i] += eligibilityTrace[i]
        
        #multiply by learning rate, reward
        grads = map(lambda x: (r-self.baseline) * x, grads)
        
        self.optimizer.apply_gradients(zip(
            grads,
            self.trainable_weights
        )) #Update weights. This function negates the gradient! But we've already inverted the characteristic eligibility (see below).
        return {"loss": (r-self.baseline)}
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch, callbacks=[]):
        def characteristic_eligibilities(s, a):
            def lng(a, p):
                return -tfp.distributions.Categorical(probs=p).log_prob(a) #apply_gradients inverts the gradient, so it must be inverted here as well
            with tf.GradientTape() as tape:
                return tape.gradient(lng(a,self(s)[0]), self.trainable_weights)
        #epoch ends, reset env, observation, & reward
        if endOfEpoch:
            self.epsilon *= self.epsilonDecay #epsilon decay
            #train model
            #zip observations & rewards, pass to fit
            eligibilityTraces = []
            
            #calculate characteristic eligibilities
            for i in range(len(observationsThisEpoch)):
                eligibilityTraces.append(characteristic_eligibilities(observationsThisEpoch[i], actionsThisEpoch[i]))
            
            self.train_step(eligibilityTraces, float(sum(rewardsThisEpoch)))
#from Function Optimization Using Connectionist Reinforcement Learning Algorithms, Williams & Peng, 1991.
#TODO if there's time, rework this to use a loss function (currently it doesn't due to my poor understanding of the topic at the time of implementation)
class REINFORCE_MENTAgent(AbstractPolicyAgent):
    def __init__(self, learningRate, actionSpace, hiddenLayers, validActions=None, epsilon=0, epsilonDecay=1, discountRate=1, baseline=0, entropyWeight=1):
        super().__init__(learningRate, actionSpace, hiddenLayers, validActions, epsilon, epsilonDecay)
        self.discountRate = discountRate
        self.baseline = baseline
        self.entropyWeight = entropyWeight
    def train_step(self, eligibilityTraces, r):
        #calculate sum of eligibility traces
        grads = [None] * len(eligibilityTraces[0])
        for eligibilityTrace in eligibilityTraces:
            for i in range(len(eligibilityTrace)):
                if grads[i] is None:
                    grads[i] = eligibilityTrace[i]
                else:
                    grads[i] += eligibilityTrace[i]
        
        #multiply by learning rate, reward
        grads = map(lambda x: (r-self.baseline) * x, grads)
        
        #Update weights. This function negates the gradient! But we've already negated the characteristic eligibility (see below).
        self.optimizer.apply_gradients(zip(
            grads,
            self.trainable_weights
        ))
        return {"loss": (self.baseline-r)}
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch, callbacks=[]):
        def lng(a, p): #probability mass function, it DOES make sense to calculate this all at once, says so in the paper
            return -tfp.distributions.Categorical(probs=p).log_prob(a)
        def characteristic_eligibilities(s, a):
            with tf.GradientTape() as tape:
                return tape.gradient(lng(a,self(s)[0]), self.trainable_weights)
        def entropy(ss):
            entropies = []
            for s in ss: #sum up the entropy of each state
                entropy = 0
                p = self(s)[0]
                for a in range(len(self.actionSpace)):
                    entropy-= p[a] + tf.math.log(p[a])
                entropies.append(entropy)
            return sum(entropies)/len(entropies)
        #epoch ends, reset env, observation, & reward
        if endOfEpoch:
            self.epsilon *= self.epsilonDecay #epsilon decay
            #train model
            #zip observations & rewards, pass to fit
            eligibilityTraces = []
            
            #calculate characteristic eligibilities
            for i in range(len(observationsThisEpoch)):
                eligibilityTraces.append(characteristic_eligibilities(observationsThisEpoch[i], actionsThisEpoch[i]))
            
            self.train_step(eligibilityTraces, sum(rewardsThisEpoch) + self.entropyWeight*entropy(observationsThisEpoch))

#value-based
#Replay method from Playing Atari with Deep Reinforcement Learning, Mnih et al (Algorithm 1).
class DQNAgent(AbstractQAgent):
    def __init__(self, learningRate, actionSpace, hiddenLayers, validActions=None, epsilon=0, epsilonDecay=1, discountRate=1, replayMemoryCapacity=1000, replayFraction=5):
        super().__init__(learningRate, actionSpace, hiddenLayers, validActions, epsilon, epsilonDecay)
        self.discountRate = discountRate
        self.replayMemoryS1s = []
        self.replayMemoryA1s = []   
        self.replayMemoryRs  = []
        self.replayMemoryS2s = []
        self.replayMemoryCapacity = replayMemoryCapacity
        self.replayFraction = replayFraction
    def train_step(self, data):
        def l(): #from atari paper
            s1,a1,r,s2 = data
            q2 = self.discountRate*tf.reduce_max(self(s2)) #estimated q-value for on-policy action for s2

            #TODO try fixed weights. create a duplicate model, use it to estimate q2, don't update its weights until end of fit()

            q1 = self(s1)[0][a1] #estimated q-value for (s,a) yielding r
            return (r+q2-q1)*(r+q2-q1) #calculate error between prediction and (approximated) label
        self.optimizer.minimize(l, self.trainable_weights)
        return {"loss": l()}
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch, callbacks=[]):
        if len(observationsThisEpoch)>1: #if we have a transition to add
            self.train_step((observationsThisEpoch[-2], actionsThisEpoch[-2], rewardsThisEpoch[-2], observationsThisEpoch[-1]))
            #add the transition
            self.replayMemoryS1s.append(observationsThisEpoch[-2])
            self.replayMemoryA1s.append(actionsThisEpoch[-2])
            self.replayMemoryRs.append(rewardsThisEpoch[-2])
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
                for i in sample:
                    miniBatchAs.append(self.replayMemoryA1s[i])
                    miniBatchRs.append(self.replayMemoryRs[i])
                    miniBatchS2s.append(self.replayMemoryS2s[i])
                dataset = tf.data.Dataset.from_tensor_slices((miniBatchS1s, miniBatchAs, miniBatchRs, miniBatchS2s))
                self.fit(dataset, batch_size=int(self.replayMemoryCapacity/(self.replayFraction*100)), callbacks=callbacks) #train on the minibatch
#This is similar to DQN, but learns on-policy.
class SARSAAgent(AbstractQAgent):
    def __init__(self, learningRate, actionSpace, hiddenLayers, validActions=None, epsilon=0, epsilonDecay=1, discountRate=1):
        super().__init__(learningRate, actionSpace, hiddenLayers, validActions, epsilon, epsilonDecay)
        self.discountRate = discountRate
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
        if len(observationsThisEpoch)>1: #if we have a transition to add
            self.train_step((observationsThisEpoch[-2], actionsThisEpoch[-2], rewardsThisEpoch[-2], observationsThisEpoch[-1],actionsThisEpoch[-1]))

#From Actor-Critic Algorithms, Konda & Tsitsiklis, NIPS 1999.
class ActorCriticAgent(AbstractActorCriticAgent):
    def __init__(self, learningRate, actionSpace, hiddenLayers, validActions=None, epsilon=0, epsilonDecay=1, discountRate=1, replayMemoryCapacity=1000, replayFraction=5, entropyWeight=.1):
        super().__init__(learningRate, actionSpace, validActions, epsilon, epsilonDecay, discountRate, entropyWeight)
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
        with tf.GradientTape(persistent=True) as tape:
            s1,a,r,s2 = data #unpack transition

            #calculate critic loss (unlike A2C, removing this step causes the grads to go NaN)
            q2 = self.discountRate*tf.reduce_max(self(s2)[0][len(self.actionSpace):]) #estimated q-value for on-policy action for s2
            q1 = self(s1)[0][len(self.actionSpace):][a] #estimated q-value for (s,a) yielding r
            l1 = (r+q2-q1)**2 #squared error between prediction and (approximated) label.

            #calculate actor loss
            #l2 = tanh(Q(s,a)) * ln(q(s,a))
            l2 =  -tf.math.log(tf.nn.softmax(self(s1)[0][:len(self.actionSpace)])[a]) #apply gradients inverts the gradient, so it must be inverted here as well
        grads = tape.gradient(l1, self.trainable_weights)
        actorGrads = tape.gradient(l2, self.trainable_weights) 
        for i in range(len(grads)): #combine actor and critic grads
            grads[i] += tf.nn.tanh(self(s1)[0][len(self.actionSpace):][a]) * actorGrads[i]
            #tanh, because the paper (actor-critic NIPS 1999) seems to claim that the actor feedback needs to be normalised
            #in some way, and without it my critic Q-values and gradients keep exploding

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights)) #this function negates the gradient!
    
        return {"loss": 0.0} #TODO
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch, callbacks=[]): 
        def entropy(Ss):
            entropies = []
            for s in Ss: #sum up the entropy of each state
                entropy = 0
                p = tf.nn.softmax(self(s)[0][:len(self.actionSpace)])
                for a in range(len(self.actionSpace)):
                    entropy-= p[a] + tf.math.log(p[a])
                entropies.append(entropy)
            return sum(entropies)/len(entropies)
        if len(observationsThisEpoch)>1: #if we have a transition to add
            #add the transition
            self.replayMemoryS1s.append(observationsThisEpoch[-2])
            self.replayMemoryA1s.append(actionsThisEpoch[-2])
            self.replayMemoryRs.append(rewardsThisEpoch[-2])
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
                e = entropy(miniBatchS1s) * self.entropyWeight
                dataset = tf.data.Dataset.from_tensor_slices((miniBatchS1s, miniBatchAs, [r+e for r in miniBatchRs], miniBatchS2s))
                self.fit(dataset) #train on the minibatch
#Synchronous verson of A3C, From Asynchronous Methods for Deep Reinforcement Learning.
class AdvantageActorCriticAgent(AbstractActorCriticAgent):
    def __init__(self, learningRate, actionSpace, hiddenLayers, validActions=None, epsilon=0, epsilonDecay=1, discountRate=1, entropyWeight=.1, tMax=1000):
        super().__init__(learningRate, actionSpace, validActions, epsilon, epsilonDecay, discountRate, entropyWeight)
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
        with tf.GradientTape() as tape:
            s1,a,adv = data #unpack transition
            lng = -tfp.distributions.Categorical(probs=tf.nn.softmax(self(s1)[0][:-1])).log_prob(a) #characteristic eligibility. apply_gradients inverts the gradient, so it must be inverted here as well

        grads = tape.gradient(lng, self.trainable_weights)
        for i in range(len(grads)): #combine actor and critic grads
            grads[i]*=tf.nn.tanh(adv)
            #tanh, because the paper (actor-critic NIPS 1999) seems to claim that the actor feedback needs to be normalised
            #in some way, and without it my critic Q-values and gradients keep exploding
            #not sure if this is necessary w/ A2C, they don't do in the original paper, it doesn't seem to affect the results
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights)) #this function negates the gradient!
        return {"loss": 0.0} #TODO
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch, callbacks=[]):
        def entropy(Ss):
            entropies = []
            for s in Ss: #sum up the entropy of each state
                entropy = 0
                p = tf.nn.softmax(self(s)[0][:-1])
                for a in range(len(self.actionSpace)):
                    entropy-= p[a] + tf.math.log(p[a])
                entropies.append(entropy)
            return sum(entropies)/len(entropies)
        def advantage(Ss,Rs): #calculate Rt + V(st+1...k) + V(st+k), for all states that haven't been trained on yet (all steps since the last t divisible by tMax)
            m = (len(observationsThisEpoch)%self.tMax)
            k = self.tMax if m==0 else m
            t = len(observationsThisEpoch) - k
            advantage = 0
            for i in range(k):
                advantage += (self.discountRate**i*float(Rs[t+i])) + (self.discountRate**k*self(Ss[t+k-1])[0][-1]) - (self(Ss[t])[0][-1]) #advantage formula from Async Methods for DRL
            return advantage
        if endOfEpoch:
            self.epsilon *= self.epsilonDecay #epsilon decay
        if (len(observationsThisEpoch)%self.tMax==0) or endOfEpoch:
            data = (observationsThisEpoch[-1], actionsThisEpoch[-1], advantage(observationsThisEpoch, rewardsThisEpoch) + entropy(observationsThisEpoch)*self.entropyWeight)
            self.train_step(data) #train on the minibatch
#from Proximal Policy Optimisation (???) TODO
class PPOAgent(AbstractActorCriticAgent):
    def __init__(self, learningRate, actionSpace, hiddenLayers, validActions=None, epsilon=0, epsilonDecay=1, discountRate=1, entropyWeight=.1, tMax=1000, interval=0.2):
        super().__init__(learningRate, actionSpace, validActions, epsilon, epsilonDecay, discountRate, entropyWeight)
        self.tMax = tMax
        self.interval = interval
        self.modelLayers = []
        self.modelLayers.extend(hiddenLayers)
        self.modelLayers.append(layers.Dense(len(actionSpace)+1)) #last output is the state V-value, all others are the action probs 
        self.compile(
            optimizer=tf.optimizers.Adam(learning_rate=self.learningRate),
            metrics="loss"
        )
    def train_step(self, data):
        def l():
            s,a,pOld,adv = data
            #calc rt(θ)
            pNew = self(s)[0][a]
            rt = pNew/pOld
            return tf.minimum(rt * adv, tf.clip_by_value(rt, 1-self.interval, 1+self.interval) * adv) #lower bound, clip and stuff
        self.optimizer.minimize(l, self.trainable_weights)
        return {"loss": l()}
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch, callbacks=[]):
        def advantage(Ss,Rs,t): #advantage formula from Async Methods for DRL. calculate Rt + V(st+1...k) + V(st+k), for all states that haven't been trained on yet (all steps since the last t divisible by tMax)
            advantage = (self(Ss[t])[0][-1]) - (self(Ss[-1])[0][-1])
            for i in range(len(Ss)-t):
                advantage += self.discountRate**i * Rs[t+i]
            return tf.nn.tanh(advantage)
        def entropy(Ss):
            entropies = []
            for s in Ss: #sum up the entropy of each state
                entropy = 0
                p = tf.nn.softmax(self(s)[0][:-1])
                for a in range(len(self.actionSpace)):
                    entropy-= p[a] + tf.math.log(p[a])
                entropies.append(entropy)
            return sum(entropies)/len(entropies)
        if (len(observationsThisEpoch)%self.tMax==0) or endOfEpoch:
            m = (len(observationsThisEpoch)%self.tMax)
            trajectoryLength = self.tMax if m==0 else m
            e = entropy(observationsThisEpoch)*self.entropyWeight
            advantages = []
            for i in range(trajectoryLength,0,-1): #calculate new advantages
                advantages.append(advantage(observationsThisEpoch, rewardsThisEpoch, len(observationsThisEpoch)-1-i) + e)
            observationsSlice = observationsThisEpoch[-trajectoryLength:]
            actionsSlice = actionsThisEpoch[-trajectoryLength:]
            dataset = tf.data.Dataset.from_tensor_slices((
                observationsSlice,
                actionsSlice,
                [self(s)[0][a] for s,a in zip(observationsSlice, actionsSlice)], #calculate pOlds, these are the action probs under the policy that generated the transitions
                advantages
            ))
            self.fit(dataset) #train on the minibatch
        if endOfEpoch:
            self.epsilon *= self.epsilonDecay #epsilon decay
            self.advantages = []