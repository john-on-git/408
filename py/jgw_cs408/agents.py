import tensorflow as tf
from keras import Model
from keras import layers as layers
import numpy as np
import random
import tensorflow_probability as tfp
import math

#TODO
    #all agents need to implement blocking of invalid actions. This also needs to be applied to the training logic, i.e. called by q()
    #remove flatten layer?

class RandomAgent():
    def __init__(self, actionSpace):
        self.epsilon = 1
        self.actionSpace = actionSpace
        pass
    def act(self, _):
        return random.choice(self.actionSpace)
    def handleStep(self, _, __, ___, ____):
        pass
    def save_weights(self, path, overwrite):
        pass

class QAgent(Model):
    def __init__(self, learningRate, actionSpace, hiddenLayers=[layers.Dense(16, activation=tf.nn.relu),layers.Dense(32, activation=tf.nn.relu)], epsilon=0, epsilonDecay=1):
        super().__init__()
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.actionSpace = actionSpace

        #initLayers
        self.modelLayers = []
        self.modelLayers.append(layers.Flatten())
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
        self.epsilon *= self.epsilonDecay
        if random.random()<self.epsilon: #chance to act randomly
            return random.choice(self.actionSpace)
        else:
            return int(tf.argmax(self(s)[0])) #follow greedy policy
    def train_step(self, x):
        pass
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch, callbacks=[]):
        pass
class PolicyAgent(Model):
    def __init__(self, learningRate, actionSpace, hiddenLayers=[layers.Dense(16, activation=tf.nn.relu),layers.Dense(32, activation=tf.nn.relu)], epsilon=0, epsilonDecay=1):
        super().__init__()
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.actionSpace = actionSpace

        #initLayers
        self.modelLayers = []
        self.modelLayers.append(layers.Flatten())
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
        self.epsilon *= self.epsilonDecay
        if random.random()<self.epsilon: #chance to act randomly
            return random.choice(self.actionSpace)
        else:
            return int(tf.random.categorical(logits=self(s),num_samples=1))
    def train_step(self, x):
        pass
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch, callbacks=[]):
        pass

class REINFORCEAgent(PolicyAgent):
    def __init__(self, learningRate, actionSpace, hiddenLayers=[layers.Dense(16, activation=tf.nn.relu),layers.Dense(32, activation=tf.nn.relu)], epsilon=0, epsilonDecay=1, discountRate=0, baseline=0):
        super().__init__(learningRate, actionSpace, hiddenLayers, epsilon, epsilonDecay)
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
        )) #update weights
        return {"loss": (self.baseline-r)}
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch, callbacks=[]):
        def characteristic_eligibilities(s, a):
            def lng(a, p): #probability mass function, it DOES make sense to calculate this all at once, says so in the paper
                #The model only converges if we invert the gradient but I have no idea why. 💀
                #I think it's because apply_gradients is intended to apply gradient w/r to a loss function so it subtracts it by default, but StackEx posters claim that this isn't the case.
                #(could also be inverted elsewhere but I think doing it here is clearest)
                return -tfp.distributions.Categorical(p).log_prob(a)
            with tf.GradientTape() as tape:
                return tape.gradient(lng(a,self(s)), self.trainable_weights)
        #epoch ends, reset env, observation, & reward
        if endOfEpoch:
            #train model
            #zip observations & rewards, pass to fit
            eligibilityTraces = []
            
            #calculate characteristic eligibilities
            for i in range(len(observationsThisEpoch)):
                eligibilityTraces.append(characteristic_eligibilities(observationsThisEpoch[i], actionsThisEpoch[i]))
            
            self.train_step(eligibilityTraces, float(sum(rewardsThisEpoch)))

class REINFORCE_MENTAgent(PolicyAgent):
    def __init__(self, learningRate, actionSpace, hiddenLayers=[layers.Dense(16, activation=tf.nn.relu),layers.Dense(32, activation=tf.nn.relu)], epsilon=0, epsilonDecay=1, discountRate=0, baseline=0):
        super().__init__(actionSpace, hiddenLayers, epsilon, epsilonDecay)
        self.discountRate = discountRate
        self.baseline = baseline
        self.learningRate = learningRate
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
        )) #update weights
        return {"loss": (self.baseline-r)}
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch, callbacks=[]):
        def lng(a, p): #probability mass function, it DOES make sense to calculate this all at once, says so in the paper
            #The model only converges if we invert the gradient but I have no idea why. 💀
            #I think it's because apply_gradients is intended to apply gradient w/r to a loss function so it subtracts it by default, but StackEx posters claim that this isn't the case.
            #(could also be inverted elsewhere but I think doing it here is clearest)
            return -tfp.distributions.Categorical(p).log_prob(a)
        def characteristic_eligibilities(s, a):
            with tf.GradientTape() as tape:
                return tape.gradient(lng(a,self(s)), self.trainable_weights)
        def entropy(ss):
            entropies = []
            for s in ss: #sum up the entropy of each state
                entropy = 0
                p = self(s)
                for a in range(len(self.actionSpace)):
                    entropy-= tfp.distributions.Categorical(p).prob(a) + tfp.distributions.Categorical(p).log_prob(a)
                entropies.append(entropy)
            return sum(entropies)/len(entropies)
        #epoch ends, reset env, observation, & reward
        if endOfEpoch:
            #train model
            #zip observations & rewards, pass to fit
            eligibilityTraces = []
            
            #calculate characteristic eligibilities
            for i in range(len(observationsThisEpoch)):
                eligibilityTraces.append(characteristic_eligibilities(observationsThisEpoch[i], actionsThisEpoch[i]))
            
            self.train_step(eligibilityTraces, sum(rewardsThisEpoch) + self.entropyWeight*entropy(observationsThisEpoch))

#Replay method from Playing Atari with Deep Reinforcement Learning, Mnih et al (Algorithm 1).
class DQNAgent(QAgent):
    def __init__(self, learningRate, actionSpace, hiddenLayers=[layers.Dense(16, activation=tf.nn.relu),layers.Dense(32, activation=tf.nn.relu)], epsilon=0, epsilonDecay=1, discountRate=0, baseline=0, replayMemoryCapacity=0, replayFraction=5, entropyWeight=1):
        super().__init__(learningRate, actionSpace, hiddenLayers, epsilon, epsilonDecay)
        self.discountRate = discountRate
        self.baseline = baseline
        self.replayMemoryS1s = []
        self.replayMemoryA1s = []   
        self.replayMemoryRs  = []
        self.replayMemoryS2s = []
        self.replayMemoryCapacity = replayMemoryCapacity
        self.replayFraction = replayFraction
        self.entropyWeight=entropyWeight
    def train_step(self, x):
        def l(s1,a1,r,s2): #from atari paper    
            q2 = self.discountRate*tf.reduce_max(self(s2)) #estimated q-value for on-policy action for s2

            #TODO try fixed weights. create a duplicate model, use it to estimate q2, don't update its weights until end of fit()

            q1 = self(s1)[0][a1] #estimated q-value for (s,a) yielding r
            return (r+q2-q1)*(r+q2-q1) #tf.math.squared_difference(r+q2, q1) #calculate error between prediction and (approximated) label

        s1, a, r, s2 = x
        self.optimizer.minimize(lambda: l(s1,a,r,s2), self.trainable_weights)
        return {"loss": l(s1,a,r,s2)}
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch, callbacks=[]):
        def entropy(ss):
            if len(ss)>=2:
                return 0
            else:
                actionCounts = [0] * len(self.actionSpace)
                entropies = []
                for s in ss: #sum up the entropy of each state
                    a = tf.argmax(self(s)[0])
                    if actionCounts[a] != 0 and sum(actionCounts) != 0:
                        prob = actionCounts[a]/sum(actionCounts)
                        entropies.append(-(prob+math.log(prob)))
                    actionCounts[a]+=1
                return sum(entropies)/len(entropies)
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
                policyEntropy = self.entropyWeight*entropy(random.sample(self.replayMemoryS1s, min(len(self.replayMemoryS1s), int(self.replayMemoryCapacity/self.replayFraction))))
                #build the minibatch
                miniBatchS1s = []
                miniBatchAs  = []
                miniBatchRs  = []
                miniBatchS2s = []
                for i in random.sample(range(len(self.replayMemoryS1s)), min(len(self.replayMemoryS1s), int(self.replayMemoryCapacity/self.replayFraction))):
                    miniBatchS1s.append(self.replayMemoryS1s[i])
                    miniBatchAs.append(self.replayMemoryA1s[i])
                    miniBatchRs.append(self.replayMemoryRs[i]+policyEntropy)
                    miniBatchS2s.append(self.replayMemoryS2s[i])
                dataset = tf.data.Dataset.from_tensor_slices((miniBatchS1s, miniBatchAs, miniBatchRs, miniBatchS2s))
                self.fit(dataset, batch_size=int(self.replayMemoryCapacity/(self.replayFraction*100)), callbacks=callbacks) #train on the minibatch

#TODO modify. The use of replay memory makes this not SARSA, and unlikely to improve.
#same as above but SARSA instead
class SARSAAgent(Model):
    def __init__(self, learningRate, actionSpace, hiddenLayers=[layers.Dense(16, activation=tf.nn.relu),layers.Dense(32, activation=tf.nn.relu)], entropyWeight=1, discountRate=0, baseline=0, epsilon=0, epsilonDecay=1, replayMemoryCapacity=0, replayFraction=5):
        super().__init__(learningRate, actionSpace, hiddenLayers, epsilon, epsilonDecay)
        self.discountRate = discountRate
        self.baseline = baseline
        self.replayMemoryS1s = []
        self.replayMemoryA1s = []
        self.replayMemoryRs  = []
        self.replayMemoryS2s = []
        self.replayMemoryA2s = []
        self.replayMemoryCapacity = replayMemoryCapacity
        self.replayFraction = replayFraction
        self.entropyWeight=entropyWeight
    def train_step(self, x):
        @tf.function
        def l(s1,a1,r,s2,a2): #from atari paper
            q2 = self.discountRate*self(s2)[0][a2] #estimated q-value for on-policy action for s2
            q1 = self(s1)[0][a1] #estimated q-value for (s,a) yielding r
            return tf.math.squared_difference(r+q2, q1)
        s1,a1,r,s2,a2 = x
        self.optimizer.minimize(lambda: l(s1,a1,r,s2,a2), self.trainable_weights)
        return {"loss": l(s1,a1,r,s2,a2)}
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch, callbacks=[]):
        if len(observationsThisEpoch)>1: #if we have a transition to add
            self.train_step((observationsThisEpoch[-2], actionsThisEpoch[-2], rewardsThisEpoch[-2], observationsThisEpoch[-1], actionsThisEpoch[-1]))
            #add the transition
            self.replayMemoryS1s.append(observationsThisEpoch[-2])
            self.replayMemoryA1s.append(actionsThisEpoch[-2])
            self.replayMemoryRs.append(rewardsThisEpoch[-2])
            self.replayMemoryS2s.append(observationsThisEpoch[-1])
            self.replayMemoryA2s.append(actionsThisEpoch[-1])
            if len(self.replayMemoryS1s)>self.replayMemoryCapacity: #if this puts us over capacity remove the oldest transition to put us back under cap
                self.replayMemoryS1s.pop(0)
                self.replayMemoryA1s.pop(0)
                self.replayMemoryRs.pop(0)
                self.replayMemoryS2s.pop(0)
                self.replayMemoryA2s.pop(0)
            
            if endOfEpoch:
                #build the minibatch
                miniBatchS1s = []
                miniBatchAs  = []
                miniBatchRs  = []
                miniBatchS2s = []
                miniBatchA2s = []
                
                for i in random.sample(range(len(self.replayMemoryS1s)), min(len(self.replayMemoryS1s), int(self.replayMemoryCapacity/self.replayFraction))):
                    miniBatchS1s.append(self.replayMemoryS1s[i])
                    miniBatchAs.append(self.replayMemoryA1s[i])
                    miniBatchRs.append(self.replayMemoryRs[i])
                    miniBatchS2s.append(self.replayMemoryS2s[i])
                    miniBatchA2s.append(self.replayMemoryA2s[i])
                dataset = tf.data.Dataset.from_tensor_slices((miniBatchS1s, miniBatchAs, miniBatchRs, miniBatchS2s, miniBatchA2s))
                self.fit(dataset, batch_size=int(self.replayMemoryCapacity/(self.replayFraction*100)), callbacks=callbacks) #train on the minibatch

class Actor(PolicyAgent):
    def __init__(self, learningRate, actionSpace):
        super().__init__()
        self.modelLayers = [
            layers.Flatten(),
            layers.Dense(16, activation=tf.nn.relu),
            layers.Dense(32, activation=tf.nn.relu),
            layers.Dense(len(actionSpace), tf.nn.softmax)
        ]
        self.compile(
            optimizer=tf.optimizers.Adam(
                learning_rate = learningRate
            ),
            metrics=["loss"]
        )
    def call(self, s):
        for layer in self.layers:
            s = layer(s)
        return s
    def train_step(self, x):
        def l(s,a,v):
            return v * tfp.distributions.Categorical(self(s)[0]).log_prob(a)
        s,a,v = x #where v is the critic's feedback
        self.optimizer.minimize(lambda: l(s,a,v), self.trainable_weights)
        return {"loss": l(s,a,v)} #idk if taking the average here makes sense
class Critic(Model):
    def __init__(self, learningRate, actionSpace, discountRate):
        super().__init__()
        self.discountRate = discountRate
        self.modelLayers = [
            layers.Flatten(),
            layers.Dense(16, activation=tf.nn.relu),
            layers.Dense(32, activation=tf.nn.relu),
            layers.Dense(len(actionSpace))
        ]
        self.compile(
            optimizer=tf.optimizers.Adam(
                learning_rate = learningRate
            ),
            metrics=["loss"]
        )
    def call(self, s):
        for layer in self.layers:
            s = layer(s)
        return s
    def train_step(self, x):
        def l(s1,a1,r,s2):
            #there's a function phi that transforms (s,a) into the critic input
            #len(critic logits) = len(actor.trainable_weights)
            q2 = self.discountRate*tf.reduce_max(self(s2)) #estimated q-value for on-policy action for s2
            q1 = self(s1)[0][a1] #estimated q-value for (s,a) yielding r
            return (r+q2-q1)*(r+q2-q1) #tf.math.squared_difference(r+q2, q1) #calculate error between prediction and (approximated) label
        s1,a,r,s2 = x
        self.optimizer.minimize(lambda: l(s1,a,r,s2), self.trainable_weights)
        return {"loss": l(s1,a,r,s2)} #idk if taking the average here makes sense
class ActorCriticAgent(Model):
    def __init__(self, learningRate, actionSpace, entropyWeight=1, discountRate=0, baseline=0, epsilon=0, epsilonDecay=1, replayMemoryCapacity=0, replayFraction=5, validActions = None):
        super().__init__()
        self.discountRate = discountRate
        self.baseline = baseline
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.learningRate = learningRate
        self.actionSpace = actionSpace
        self.replayMemoryS1s = []
        self.replayMemoryA1s = []
        self.replayMemoryRs  = []
        self.replayMemoryS2s = []
        self.replayMemoryA2s = []
        self.replayMemoryCapacity = replayMemoryCapacity
        self.replayFraction = replayFraction
        self.entropyWeight=entropyWeight
        self.validActions = validActions if validActions is not None else (lambda s: actionSpace) #Callable that returns the valid actions for a state. Defaults to entire action space.

        #initLayers
        self.discountRate = np.float32(discountRate)
        self.actor = Actor(learningRate, actionSpace)
        self.critic = Critic(learningRate, actionSpace, discountRate)
        self.compile()
    def act(self, s):
        self.epsilon *= self.epsilonDecay #epsilon decay
        validActions = self.validActions(s) #get valid actions
        if random.random()<self.epsilon: #chance to act randomly
            return random.choice(validActions)
        else:
            probs = self.actor(s)[0]
            newProbs = [0.0] * len(probs)
            for i in validActions:
                newProbs[i] = probs[i] 
            return int(tf.argmax(probs)) #follow greedy policy
    def train_step(self, x):
        l1 = self.critic.train_step(x)
        s1,a,_,_ = x
        v = self.actor(s1)[0][a]
        l2 = self.actor.train_step((s1,a,v))
        return {"loss": (l1["loss"]+l2["loss"])/2} #idk if taking the average here makes sense
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch, callbacks=[]):
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
                self.train_step((observationsThisEpoch[-2], actionsThisEpoch[-2], rewardsThisEpoch[-2], observationsThisEpoch[-1])) #load-bearing train_step
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
                dataset = tf.data.Dataset.from_tensor_slices((miniBatchS1s, miniBatchAs, miniBatchRs, miniBatchS2s))
                self.fit(dataset) #train on the minibatch
    #warning, very dumb
    def save_weights(self,path, overwrite):
        self.actor.save_weights(path+"_actor.tf", overwrite=overwrite)
        self.critic.save_weights(path+"_critic.tf", overwrite=overwrite)
    def load_weights(self,path):
        self.actor.load_weights(path+"_actor.tf")
        self.critic.load_weights(path+"_critic.tf")

class AdvantageActorCriticAgent (Model):
    def __init__(self, learningRate, actionSpace, entropyWeight=1, discountRate=0, baseline=0, epsilon=0, epsilonDecay=1, validActions = None):
        super().__init__()
        self.discountRate = discountRate
        self.baseline = baseline
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.learningRate = learningRate
        self.actionSpace = actionSpace
        self.entropyWeight=entropyWeight
        self.validActions = validActions if validActions is not None else (lambda s: actionSpace) #Callable that returns the valid actions for a state. Defaults to entire action space.
        #initLayers
        self.discountRate = np.float32(discountRate)
        self.modelLayers = [
            layers.Flatten(),
            layers.Dense(16, activation=tf.nn.relu),
            layers.Dense(32, activation=tf.nn.relu),
            layers.Dense(len(actionSpace)+1) #the final output is the v-value of s
        ]
        self.compile()
    def call(self, observation):
        for layer in self.layers:
            observation = layer(observation)
        return observation
    def act(self, s):
        self.epsilon *= self.epsilonDecay #epsilon decay
        validActions = self.validActions(s) #get valid actions
        if random.random()<self.epsilon: #chance to act randomly
            return random.choice(validActions)
        else:
            probs = self(s)[0][0:-1]
            newProbs = [0.0] * len(probs)
            for i in validActions:
                newProbs[i] = probs[i] 
            return int(tf.argmax(probs)) #follow greedy policy
    def train_step(self, x):
        def lCritic(s1,r,s2):
            q2 = self.discountRate*self(s2)[0][-1] #estimated v-value for s2
            q1 = self(s1)[0][-1] #estimated v-value for s1
            return (r+q2-q1)*(r+q2-q1) #tf.math.squared_difference(r+q2, q1) #calculate error between prediction and (approximated) label
        def lActor(s,a,v):
            return v * tfp.distributions.Categorical(self(s)[0][:-1]).log_prob(a)
        s1,act,r,adv,s2 = x
    
        self.optimizer.minimize(lambda: lCritic(s1,r,s2), self.trainable_weights) #minimize critic
        self.optimizer.minimize(lambda: lActor(s1,act,adv), self.trainable_weights) #minimize actor
        return {"loss": 0.0} #TODO
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch, callbacks=[]):
        def advantages(Ss, Rs, tMax): #calculate advantage for all ts
            advantages = []
            advantage = 0
            n = 0
            for t in range(tMax):
                k = tMax-t
                for i in range(k):
                    n+=1
                    advantage += (self.discountRate**i*Rs[t+i]) + (self.learningRate**k*self(Ss[t+k])[0][-1]) - (self(Ss[t])[0][-1]) #advantage formula from Async Methods for DRL
                advantages.append(advantage)
            return advantages
        if endOfEpoch:
            #self.train_step((observationsThisEpoch[-2], actionsThisEpoch[-2], rewardsThisEpoch[-2], observationsThisEpoch[-1])) #load-bearing train_step
            dataset = tf.data.Dataset.from_tensor_slices((observationsThisEpoch[:-1], actionsThisEpoch[:-1], rewardsThisEpoch[:-1], advantages(observationsThisEpoch, rewardsThisEpoch, len(observationsThisEpoch)-1), observationsThisEpoch[1:]))
            self.fit(dataset) #train on the minibatch