import tensorflow as tf
from keras import Model
from keras import layers as layers
import random

class PPOModel(Model):
    def __init__(self, learningRate, discountRate, epsilonFraction):
        super().__init__()
        self.learningRate = learningRate
        self.discountRate = discountRate
        self.epsilonFraction = epsilonFraction
        self.modelLayers = [
            layers.Flatten(input_shape=(1, 5, 5)),
            layers.Dense(16, activation=tf.nn.relu),
            layers.Dense(32, activation=tf.nn.relu),
            layers.Dense(5, activation=tf.nn.sigmoid),
            layers.Softmax()
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
            return random.choice([0,1,2,3])
        else:
            a = int(tf.random.categorical(logits=self(observation),num_samples=1))
            return 4 if a>4 else a #categorical can return nCategories+1 (when spread is all zeros?)
    def train_step(self, datum):
        s, _, r = datum
        self.optimizer.minimize(lambda: self(s)*-r, self.trainable_weights)
        return {} #TODO what format is this supposed to be in, not really important imo
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch):
        #epoch ends, reset env, observation, & reward
        if endOfEpoch:
            #train model
            #zip observations & rewards, pass to fit
            avgRewardThisEpoch = [sum(rewardsThisEpoch)/len(rewardsThisEpoch)] * len(rewardsThisEpoch)            
            data = tf.data.Dataset.from_tensor_slices((observationsThisEpoch, actionsThisEpoch, avgRewardThisEpoch))
            self.fit(data)