import tensorflow as tf
from keras import Model
from keras import layers as layers
import numpy as np
import random

#based on Actor-Critic algorithms, Konda & Tsitsiklis, NIPS 1999
class DQNModel(Model):
    def __init__(self, learningRate, discountRate, epsilonFraction=20):
        super().__init__()
        self.epsilonFraction = epsilonFraction
        self.learningRate = np.float32(learningRate)
        self.discountRate = np.float32(discountRate)
        self.actorLayers = [
            #layers.ZeroPadding2D(padding=(100,100)),
            layers.Flatten(input_shape=(1, 5, 5)),
            layers.Dense(16, activation=tf.nn.relu),
            layers.Dense(32, activation=tf.nn.relu),
            layers.Dense(4, activation=tf.nn.sigmoid)
        ]
        self.criticLayers = [
            
        ]
        self.compile(
            optimizer="Adam",
            metrics="loss"
        )
    def call(self, observation):
        for layer in self.actorLayers:
            observation = layer(observation)
        return observation
    def act(self, observation):
        if random.randint(1,self.epsilonFraction)==1: #chance to act randomly
            return random.choice([0,1,2,3])
        else:
            return int(tf.argmax(tf.reshape(self(observation), shape=(4,1)))) #follow greedy policy
    def train_step(self, xs):
        return {}
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch):
        pass