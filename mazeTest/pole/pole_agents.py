import tensorflow as tf
from keras import Model
from keras import layers as layers
import numpy as np
import random
import tensorflow_probability as tfp

#chooses randomly
class RandomAgent():
    def __init__(self):
        self.epsilon = 0
        pass
    def act(self, _):
        return random.choice([0,1])
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch, callbacks=[]):
        pass
    def save_weights(self, path, overwrite):
        pass

class WeirdREINFORCEVariant(Model):
    def __init__(self, learningRate, discountRate, baseline=0, epsilon=0, epsilonDecay=1):
        super().__init__()
        self.discountRate = discountRate
        self.baseline = baseline
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.learningRate = learningRate
        self.modelLayers = [
            layers.Flatten(input_shape=(4,)),
            layers.Dense(4, activation=tf.nn.relu),
            layers.Dense(8, activation=tf.nn.relu),
            layers.Dense(2, activation=tf.nn.softmax),
        ]
        self.compile(
            optimizer=tf.optimizers.Adam(),
            metrics="loss"
        )
    def call(self, observation):
        for layer in self.modelLayers:
            observation = layer(observation)
        return observation
    def act(self, observation):
        self.epsilon*=self.epsilonDecay
        if random.random()<self.epsilon: #chance to act randomly
            return random.choice([0,1])
        else:
            return int(tf.random.categorical(logits=self(observation),num_samples=1))
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
        grads = map(lambda x: self.learningRate * (float(r)-self.baseline)*x, grads)
        
        self.optimizer.apply_gradients(zip(
            grads,
            self.trainable_weights
        )) #update weights
        return {"loss": (self.baseline-float(r))}
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch, callbacks=[]):
        def characteristic_eligibilities(s, a):
            def lng(a, p): #probability mass function, it DOES make sense to calculate this all at once, says so in the paper
                #The model only converges if we invert the gradient but I have no idea why. 💀
                #I think it's because this is intended to find the gradient w/r to a loss function so it subtracts it by default
                return -tfp.distributions.Categorical(p).log_prob(a)
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

class REINFORCEAgent(Model):
    def __init__(self, learningRate, discountRate=0, baseline=0, epsilon=0, epsilonDecay=1):
        super().__init__()
        self.discountRate = discountRate
        self.baseline = baseline
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.learningRate = learningRate
        self.modelLayers = [
            layers.Flatten(input_shape=(4,)),
            layers.Dense(4, activation=tf.nn.relu),
            layers.Dense(8, activation=tf.nn.relu),
            layers.Dense(2, activation=tf.nn.softmax),
        ]
        self.compile(
            optimizer=tf.optimizers.Adam(),
            metrics="loss"
        )
    def call(self, observation):
        for layer in self.modelLayers:
            observation = layer(observation)
        return observation
    def act(self, observation):
        self.epsilon*=self.epsilonDecay
        if random.random()<self.epsilon: #chance to act randomly
            return random.choice([0,1])
        else:
            return int(tf.random.categorical(logits=self(observation),num_samples=1))
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
        grads = map(lambda x: self.learningRate * (r-self.baseline) * x, grads)
        
        self.optimizer.apply_gradients(zip(
            grads,
            self.trainable_weights
        )) #update weights
        return {"loss": (self.baseline-r)}
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch, callbacks=[]):
        def characteristic_eligibilities(s, a):
            def lng(a, p): #probability mass function, it DOES make sense to calculate this all at once, says so in the paper
                #The model only converges if we invert the gradient but I have no idea why. 💀
                #I think it's because apply_gradients is intended to apply gradient w/r to a loss function so it subtracts it by default
                #(could also be inverted elsewhere but I think doing it here is clearest)
                return -tfp.distributions.Categorical(p).log_prob(a)
            with tf.GradientTape() as tape:
                return tape.gradient(lng(a,self(s)), self.trainable_weights)
        #epoch ends, reset env, observation, & reward
        if endOfEpoch:
            #train model
            #zip observations & rewards, pass to fit
            eligibilityTraces = []
            
            for i in range(len(observationsThisEpoch)):
                eligibilityTraces.append(characteristic_eligibilities(observationsThisEpoch[i], actionsThisEpoch[i]))
            
            self.train_step(eligibilityTraces, float(sum(rewardsThisEpoch)))

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
            layers.Flatten(input_shape=(4,)),
            layers.Dense(4, activation=tf.nn.sigmoid),
            layers.Dense(8, activation=tf.nn.sigmoid),
            layers.Dense(2, activation=None)
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
            layers.Flatten(input_shape=(4,)),
            layers.Dense(4, activation=tf.nn.sigmoid),
            layers.Dense(8, activation=tf.nn.sigmoid),
            layers.Dense(2, activation=None)
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
            layers.Flatten(input_shape=(4,)),
            layers.Dense(4, activation=tf.nn.sigmoid),
            layers.Dense(8, activation=tf.nn.sigmoid),
            layers.Dense(2, activation=None)
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

class ActorCriticAgent (Model):
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
        self.actorLayers = [
            layers.Flatten(input_shape=(1, 4)),
            layers.Dense(4, activation=tf.nn.relu),
            layers.Dense(8, activation=tf.nn.relu),
            layers.Dense(2, activation=tf.nn.softmax)
        ]
        self.criticLayers = [
            layers.Flatten(input_shape=(1, 4)),
            layers.Dense(4, activation=tf.nn.sigmoid),
            layers.Dense(8, activation=tf.nn.sigmoid),
            layers.Dense(2, activation=None)
        ]
        self.compile(
            optimizer=tf.optimizers.Adam(
                learning_rate = learningRate
            ),
            metrics=["loss"]
        )
    def call(self, s):
        return self.actor(s)
    def actor(self, s):
        for layer in self.actorLayers:
            s = layer(s)
        return s
    def critic(self, s):
        for layer in self.criticLayers:
            s = layer(s)
        return s
    def act(self, s):
        self.epsilon *= self.epsilonDecay #epsilon decay
        if random.random()<self.epsilon: #chance to act randomly
            return random.choice([0,1])
        else:
            return int(tf.argmax(self(s)[0])) #follow greedy policy
    def train_step(self, x):
        @tf.function
        def lCritic(s1,a1,r,s2):
            #there's a function phi that transforms (s,a) into the critic input
            #len(critic logits) = len(actor.trainable_weights)
            q2 = self.discountRate*tf.reduce_max(self(s2)) #estimated q-value for on-policy action for s2
            q1 = self(s1)[0][a1] #estimated q-value for (s,a) yielding r
            return (r+q2-q1)*(r+q2-q1) #tf.math.squared_difference(r+q2, q1) #calculate error between prediction and (approximated) label
        @tf.function
        def lActor(s,a):
            return self.critic(s)[a] * tfp.distributions.Categorical(self.actor(s)).log_prob(a)
        s1,a,r,s2 = x
        self.optimizer.minimize(lambda: lCritic(s1,a,r,s2), self.criticLayers.trainable_weights)
        self.optimizer.minimize(lambda: lActor(s1,a), self.actorLayers.trainable_weights)
        return {"loss": (lActor(s1,a)+lCritic(s1,a,r,s2))/2} #idk if taking the average here makes sense
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
                self.fit(dataset, batch_size=int(self.replayMemoryCapacity/(self.replayFraction*100)), callbacks=callbacks) #train on the minibatch