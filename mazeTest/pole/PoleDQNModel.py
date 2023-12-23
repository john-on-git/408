import tensorflow as tf
from keras import Model
from keras import layers as layers
import numpy as np
import random

#Replay method from Playing Atari with Deep Reinforcement Learning, Mnih et al (Algorithm 1).
class PoleDQNModel(Model):
    def __init__(self, learningRate, discountRate, replayMemoryCapacity=0, epsilonFraction=20):
        super().__init__()
        self.replayMemoryS1s = []
        self.replayMemoryAs  = []
        self.replayMemoryRs  = []
        self.replayMemoryS2s = []
        self.replayMemoryCapacity = replayMemoryCapacity
        self.epsilonFraction = epsilonFraction
        self.learningRate = np.float32(learningRate)
        self.discountRate = np.float32(discountRate)
        self.modelLayers = [
            #layers.ZeroPadding2D(padding=(100,100)),
            layers.Dense(16, input_shape=(1,4), activation=tf.nn.relu),
            layers.Dense(32, activation=tf.nn.relu),
            layers.Dense(2, activation=tf.nn.sigmoid)
        ]
        self.compile(
            optimizer="Adam",
            metrics="loss"
        )
    def call(self, observation):
        for layer in self.modelLayers:
            observation = layer(observation)
        return observation
    def act(self, observation):
        if random.randint(1,self.epsilonFraction)==1: #chance to act randomly
            return random.choice([0,1])
        else:
            return int(tf.argmax(tf.reshape(self(observation), shape=(2,1)))) #follow greedy policy
    def train_step(self, xs):
        #s1, _, r, s2, _ = x
        s1 = xs[0]
        a  = xs[1]
        r  = xs[2]
        s2 = xs[3]
        def l(s1,a,r,s2): #from atari paper
            yi = (r+self.discountRate*tf.reduce_max(self(s2))) #approximation of the actual Q-value: (real reward) plus (the model's prediction for the sum of future rewards)
            x = yi-tf.reshape(self(s1), shape=(2,1))[a] #subtract the model's prediction for the sum of future rewards
            return -(x*x)
        
        self.optimizer.minimize(lambda: l(s1,a,r,s2), self.trainable_weights)
        return {}
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch):
        if len(observationsThisEpoch)>1: #if we have a transition to add
            #add the transition
            self.replayMemoryS1s.append(observationsThisEpoch[-2])
            self.replayMemoryAs.append(actionsThisEpoch[-1])
            self.replayMemoryRs.append(rewardsThisEpoch[-1])
            self.replayMemoryS2s.append(observationsThisEpoch[-1])
            if len(self.replayMemoryS1s)>self.replayMemoryCapacity: #if this puts us over capacity remove the oldest transition to put us back under cap
                self.replayMemoryS1s.pop()
                self.replayMemoryAs.pop()
                self.replayMemoryRs.pop()
                self.replayMemoryS2s.pop()
            
            if endOfEpoch:
                #build the minibatch
                miniBatchS1s = []
                miniBatchAs  = []
                miniBatchRs  = []
                miniBatchS2s = []
                
                for i in random.sample(range(len(self.replayMemoryS1s)), min(len(self.replayMemoryS1s), int(self.replayMemoryCapacity/5))):
                    miniBatchS1s.append(self.replayMemoryS1s[i])
                    miniBatchAs.append(self.replayMemoryAs[i])
                    miniBatchRs.append(self.replayMemoryRs[i])
                    miniBatchS2s.append(self.replayMemoryS2s[i])
                dataset = tf.data.Dataset.from_tensor_slices((miniBatchS1s, miniBatchAs, miniBatchRs, miniBatchS2s))
                self.fit(dataset, batch_size=int(self.replayMemoryCapacity/500)) #train on the minitbatch