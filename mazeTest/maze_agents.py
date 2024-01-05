import tensorflow as tf
from keras import Model
from keras import layers as layers
import numpy as np
import random
import tensorflow_probability as tfp
import math

class RandomAgent():
    def __init__(self):
        self.epsilon = 0
        pass
    def act(self, _):
        return random.choice([0,1,2,3,4])
    def handleStep(self, _, __, ___, ____):
        pass
    def save_weights(self, path, overwrite):
        pass
class REINFORCEAgent(Model):
    def __init__(self, learningRate, discountRate, baseline=0):
        super().__init__()
        self.discountRate = discountRate
        self.baseline = baseline
        self.modelLayers = [
            layers.Flatten(input_shape=(5,5)),
            layers.Dense(16, activation=tf.nn.relu),
            layers.Dense(32, activation=tf.nn.relu),
            layers.Dense(4, activation=tf.nn.softmax),
        ]
        self.compile(
            optimizer=tf.optimizers.Adam(learning_rate=learningRate),
            metrics="loss"
        )
    def call(self, observation):
        for layer in self.modelLayers:
            observation = layer(observation)
        return observation
    def act(self, observation):
        return int(tf.random.categorical(logits=self(observation),num_samples=1))
    def train_step(self, eligibilityTraces, r):
        #calculate average eligibility trace
        grads = [None] * len(eligibilityTraces[0])
        for eligibilityTrace in eligibilityTraces:
            for i in range(len(eligibilityTrace)):
                if grads[i] is None:
                    grads[i] = eligibilityTrace[i]
                else:
                    grads[i] += eligibilityTrace[i]
        
        #multiply by reward
        grads = map(lambda x: x*(float(r)-self.baseline), grads)
        
        self.optimizer.apply_gradients(zip(
            grads,
            self.trainable_weights
        )) #update weights
        return {"loss": (self.baseline-float(r))}
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch, callbacks=[]):
        def characteristic_eligibilities(s, a):
            def lng(a, p): #probability mass function, it DOES make sense to calculate this all at once, says so in the paper
                return -tfp.distributions.Categorical(p).log_prob(a) #negative'd because we're minimizing
            with tf.GradientTape() as tape:
                return tape.gradient(lng(a,self(s)), self.trainable_weights)
        def sumOfDiscountedAndNormalizedFutureRewards(discountRate, futureRewards):
            def discount(discount, rs):
                for i in range(len(rs)):
                    rs[i] = rs[i] * discount
                    discount*=discount
                return rs
            return sum(discount(discountRate, futureRewards))
        #epoch ends, reset env, observation, & reward
        if endOfEpoch:
            #train model
            #zip observations & rewards, pass to fit
            eligibilityTraces = []
            
            for i in range(len(observationsThisEpoch)):
                eligibilityTraces.append(characteristic_eligibilities(observationsThisEpoch[i], actionsThisEpoch[i]))
                self.train_step(eligibilityTraces, sumOfDiscountedAndNormalizedFutureRewards(self.discountRate, rewardsThisEpoch[i:]))

class MonteCarloAgent(Model):
    def __init__(self, learningRate, discountRate, replayMemoryCapacity=0, replayFraction=5, epsilon=0, epsilonDecay=1):
        super().__init__()
        self.replayMemorySs = []
        self.replayMemoryAs  = []
        self.replayMemoryRs  = []
        self.replayMemoryCapacity = replayMemoryCapacity
        self.replayFraction = replayFraction
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.discountRate = np.float32(discountRate)
        self.modelLayers = [
            layers.Flatten(input_shape=(5,5)),
            layers.Dense(16, activation=tf.nn.sigmoid),
            layers.Dense(32, activation=tf.nn.sigmoid),
            layers.Dense(2)
        ]
        self.compile(
            optimizer=tf.optimizers.Adam(learning_rate=learningRate),
            metrics="loss",
        )
    def call(self, observation):
        for layer in self.modelLayers:
            observation = layer(observation)
        return observation
    def act(self, s):
        self.epsilon*=self.epsilonDecay
        if random.random()<self.epsilon: #chance to act randomly
            return random.choice([0,1])
        else:
            return int(tf.argmax(self(s)[0])) #follow greedy policy
    def train_step(self, x):
        s, a, r = x
        @tf.function
        def l(s,a,r): #from atari paper
            x = r-self(s)[0][a]
            return (x*x)
        
        self.optimizer.minimize(lambda: l(s,a,r), self.trainable_weights)
        return {"loss":l(s,a,r)}
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch, callbacks=[]):
        def sumOfDiscountedFutureRewards(discountRate, futureRewards):
            sumOfDiscountedFutureRewards = futureRewards[0]
            for i in range(1, len(futureRewards)):
                sumOfDiscountedFutureRewards += (futureRewards[i] * discountRate)
                discountRate*=discountRate
            return np.float32(sumOfDiscountedFutureRewards)
        if endOfEpoch:
            #add transitions
            for i in range(len(observationsThisEpoch)):
                self.replayMemorySs.append(observationsThisEpoch[i])
                self.replayMemoryAs.append(actionsThisEpoch[i])
                self.replayMemoryRs.append(sumOfDiscountedFutureRewards(self.discountRate, rewardsThisEpoch[i:]))
                if len(self.replayMemorySs)>self.replayMemoryCapacity: #if this puts us over capacity remove the oldest transition to put us back under cap
                    self.replayMemorySs.pop(0)
                    self.replayMemoryAs.pop(0)
                    self.replayMemoryRs.pop(0)
            
            #build the minibatch
            miniBatchS1s = []
            miniBatchAs  = []
            miniBatchRs  = []
            
            for i in random.sample(range(len(self.replayMemorySs)), min(len(self.replayMemorySs), int(self.replayMemoryCapacity/self.replayFraction))):
                miniBatchS1s.append(self.replayMemorySs[i])
                miniBatchAs.append(self.replayMemoryAs[i])
                miniBatchRs.append(self.replayMemoryRs[i])
            dataset = tf.data.Dataset.from_tensor_slices((miniBatchS1s, miniBatchAs, miniBatchRs))
            self.fit(dataset, batch_size=int(self.replayMemoryCapacity/(self.replayFraction*100)), callbacks=callbacks) #train on the minitbatch

#Replay method from Playing Atari with Deep Reinforcement Learning, Mnih et al (Algorithm 1).
class DQNAgent(Model):  
    def __init__(self, learningRate, discountRate, replayMemoryCapacity=0, replayFraction=5, epsilon=0, epsilonDecay=1):
        super().__init__()
        self.replayMemoryS1s = []
        self.replayMemoryA1s = []   
        self.replayMemoryRs  = []
        self.replayMemoryS2s = []
        self.replayMemoryCapacity = replayMemoryCapacity
        self.replayFraction = replayFraction
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.discountRate = np.float32(discountRate)
        self.modelLayers = [
            layers.Flatten(input_shape=(5,5)),
            layers.Dense(16, activation=tf.nn.sigmoid),
            layers.Dense(32, activation=tf.nn.sigmoid),
            layers.Dense(4)
        ]
        self.compile(
            optimizer=tf.optimizers.Adam(
                learning_rate = learningRate
            ),
            metrics=["loss"]
        )
    def call(self, observation):
        for layer in self.modelLayers:
            observation = layer(observation)
        return observation
    def act(self, s):
        self.epsilon *= self.epsilonDecay
        if random.random()<self.epsilon: #chance to act randomly
            return random.choice([0,1])
        else:
            return int(tf.argmax(self(s)[0])) #follow greedy policy
    def train_step(self, x):
        @tf.function
        def l(s1,a1,r,s2): #from atari paper    
            q2 = self.discountRate*tf.reduce_max(self(s2)) #estimated q-value for on-policy action for s2

            #TODO try fixed weights. create a duplicate model, use it to estimate q2, don't update its weights until end of fit()

            q1 = self(s1)[0][a1] #estimated q-value for (s,a) yielding r
            return (r+q2-q1)*(r+q2-q1) #tf.math.squared_difference(r+q2, q1) #calculate error between prediction and (approximated) label

        s1, a, r, s2 = x
        self.optimizer.minimize(lambda: l(s1,a,r,s2), self.trainable_weights)
        return {"loss": l(s1,a,r,s2)}
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
                self.fit(dataset, batch_size=int(self.replayMemoryCapacity/(self.replayFraction*100)), callbacks=callbacks) #train on the minitbatch

#same as above but SARSA instead
class SARSAAgent(Model):
    def __init__(self, learningRate, discountRate, replayMemoryCapacity=0, replayFraction=5, epsilon=0, epsilonDecay=1):
        super().__init__()
        self.replayMemoryS1s = []
        self.replayMemoryA1s = []
        self.replayMemoryRs  = []
        self.replayMemoryS2s = []
        self.replayMemoryA2s = []
        self.replayMemoryCapacity = replayMemoryCapacity
        self.replayFraction = replayFraction
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.discountRate = np.float32(discountRate)
        self.modelLayers = [
            layers.Flatten(input_shape=(5,5)),
            layers.Dense(16, activation=tf.nn.sigmoid),
            layers.Dense(32, activation=tf.nn.sigmoid),
            layers.Dense(4, activation=tf.nn.relu)
        ]
        self.compile(
            optimizer=tf.optimizers.Adam(
                learning_rate = learningRate
            ),
            metrics=["loss"]
        )
    def call(self, observation):
        for layer in self.modelLayers:
            observation = layer(observation)
        return observation
    def act(self, s):
        self.epsilon *= self.epsilonDecay #epsilon decay
        if random.random()<self.epsilon: #chance to act randomly
            return random.choice([0,1])
        else:
            return int(tf.argmax(self(s)[0])) #follow greedy policy
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