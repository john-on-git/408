import tensorflow as tf
from keras import Model
from keras import layers as layers
import numpy as np
import random
import tensorflow_probability as tfp

class RandomAgent():
    def __init__(self):
        self.epsilon = 1
        pass
    def act(self, _):
        return random.choice([0,1,2,3])
    def handleStep(self, _, __, ___, ____):
        pass
    def save_weights(self, path, overwrite):
        pass
class REINFORCEAgent(Model):
    def __init__(self, learningRate, discountRate=0, baseline=0, epsilon=0, epsilonDecay=1):
        super().__init__()
        self.discountRate = discountRate
        self.baseline = baseline
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.learningRate = learningRate
        self.modelLayers = [
            layers.Flatten(),
            layers.Dense(16, activation=tf.nn.relu),
            layers.Dense(32, activation=tf.nn.relu),
            layers.Dense(4, tf.nn.softmax)
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
            return random.choice([0,1,2,3])
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

class REINFORCE_MENTAgent(Model):
    def __init__(self, learningRate, discountRate=0, baseline=0, epsilon=0, epsilonDecay=1):
        super().__init__()
        self.discountRate = discountRate
        self.baseline = baseline
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.learningRate = learningRate
        self.modelLayers = [
            layers.Reshape(target_shape=(6,6,1)),
            layers.Conv2D(1, kernel_size=(3,3)),
            layers.Flatten(),
            layers.Dense(16, activation=tf.nn.relu),
            layers.Dense(32, activation=tf.nn.relu),
            layers.Dense(4, tf.nn.softmax)
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
            return random.choice([0,1,2,3])
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
        def lng(a, p): #probability mass function, it DOES make sense to calculate this all at once, says so in the paper
            #The model only converges if we invert the gradient but I have no idea why. 💀
            #I think it's because apply_gradients is intended to apply gradient w/r to a loss function so it subtracts it by default, but StackEx posters claim that this isn't the case.
            #(could also be inverted elsewhere but I think doing it here is clearest)
            return -tfp.distributions.Categorical(p).log_prob(a)
        def characteristic_eligibilities(s, a):
            with tf.GradientTape() as tape:
                return tape.gradient(lng(a,self(s)), self.trainable_weights)
        def averageEntropy(ss):
            totalEntropy = 0
            for s in ss: #sum up the entropy of each state
                p = self(s)
                for a in range(4):
                    totalEntropy-=lng(a,p)
            return (totalEntropy/len(ss)) #take the mean
        #epoch ends, reset env, observation, & reward
        if endOfEpoch:
            #train model
            #zip observations & rewards, pass to fit
            eligibilityTraces = []
            
            #calculate characteristic eligibilities
            for i in range(len(observationsThisEpoch)):
                eligibilityTraces.append(characteristic_eligibilities(observationsThisEpoch[i], actionsThisEpoch[i]))
            
            self.train_step(eligibilityTraces, float(sum(rewardsThisEpoch)) + averageEntropy(observationsThisEpoch))

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
            layers.Flatten(),
            layers.Dense(16, activation=tf.nn.relu),
            layers.Dense(32, activation=tf.nn.relu),
            layers.Dense(4, tf.nn.softmax)
        ]
        self.compile(
            optimizer=tf.optimizers.Adam(learning_rate=learningRate),
            metrics="loss",
        )
    def call(self, s):
        for layer in self.modelLayers:
            s = layer(s)
        return s
    def act(self, s):
        self.epsilon*=self.epsilonDecay
        if random.random()<self.epsilon: #chance to act randomly
            return random.choice([0,1,2,3])
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
            layers.Flatten(),
            layers.Dense(16, activation=tf.nn.relu),
            layers.Dense(32, activation=tf.nn.relu),
            layers.Dense(4, tf.nn.softmax)
        ]
        self.compile(
            optimizer=tf.optimizers.Adam(
                learning_rate = learningRate
            ),
            metrics=["loss"]
        )
    def call(self, s):
        for layer in self.modelLayers:
            s = layer(s)
        return s
    def act(self, s):
        self.epsilon *= self.epsilonDecay
        if random.random()<self.epsilon: #chance to act randomly
            return random.choice([0,1,2,3])
        else:
            return int(tf.argmax(self(s)[0])) #follow greedy policy
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
            layers.Conv2D(1, kernel_size=(2,2), input_shape=(6,6,1)),
            layers.Conv2D(1, kernel_size=(2,2)),
            layers.Flatten(),
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
    def call(self, s):
        for layer in self.modelLayers:
            s = layer(s)
        return s
    def act(self, s):
        self.epsilon *= self.epsilonDecay #epsilon decay
        if random.random()<self.epsilon: #chance to act randomly
            return random.choice([0,1,2,3])
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

class Actor(Model):
    def __init__(self, learningRate):
        super().__init__()
        self.modelLayers = [
            layers.Flatten(),
            layers.Dense(16, activation=tf.nn.relu),
            layers.Dense(32, activation=tf.nn.relu),
            layers.Dense(4, tf.nn.softmax)
        ]
        self.compile(
            optimizer=tf.optimizers.Adam(
                learning_rate = learningRate
            ),
            metrics=["loss"]
        )
    def call(self, s):
        for layer in self.modelLayers:
            s = layer(s)
        return s
    def train_step(self, x):
        def l(s,a,v):
            return v * tfp.distributions.Categorical(self(s)).log_prob(a)
        s1,a,v = x
        self.optimizer.minimize(lambda: l(s1,a,v), self.trainable_weights)
        return {"loss": l(s1,a,v)} #idk if taking the average here makes sense
class Critic(Model):
    def __init__(self, learningRate, discountRate):
        super().__init__()
        self.discountRate = discountRate
        self.modelLayers = [
            layers.Flatten(),
            layers.Dense(16, activation=tf.nn.relu),
            layers.Dense(32, activation=tf.nn.relu),
            layers.Dense(4, tf.nn.softmax)
        ]
        self.compile(
            optimizer=tf.optimizers.Adam(
                learning_rate = learningRate
            ),
            metrics=["loss"]
        )
    def call(self, s):
        for layer in self.modelLayers:
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
        self.actor = Actor(learningRate)
        self.critic = Critic(learningRate, discountRate)
        self.compile()
    def act(self, s):
        self.epsilon *= self.epsilonDecay #epsilon decay
        if random.random()<self.epsilon: #chance to act randomly
            return random.choice([0,1,2,3])
        else:
            return int(tf.argmax(self.actor(s)[0])) #follow greedy policy
    def train_step(self, x):
        s1,a,_,_ = x
        l1 = self.critic.train_step(x)
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