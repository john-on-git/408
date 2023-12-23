import tensorflow as tf
from keras import Model
from keras import layers as layers
import numpy as np
import tensorflow_probability as tfp

class REINFORCEModel(Model):
    def __init__(self, learningRate, baseline=0):
        super().__init__()
        self.learningRate = learningRate
        self.baseline = baseline
        self.modelLayers = [
            layers.Flatten(input_shape=(1, 5, 5)),
            layers.Dense(16, activation=tf.nn.relu),
            layers.Dense(32, activation=tf.nn.relu),
            layers.Dense(5, activation=tf.nn.sigmoid),
            layers.Softmax()
        ]
        self.compile(
            optimizer="sgd",
            metrics="loss"
        )
    def call(self, observation):
        for layer in self.modelLayers:
            observation = layer(observation)
        return observation
    def act(self, observation):
        a = int(tf.random.categorical(logits=self(observation),num_samples=1))
        assert a<=4 #categorical can return nCategories+1 (when spread is all zeros?)
        return a
    def train_step(self, eligibilityTraces, r):
        #get sum of all eligibility traces so far
        sumE = [None] * len(eligibilityTraces[0])
        for i in range(len(eligibilityTraces[0])):
            sumE[i] = eligibilityTraces[0][i]
        
        for i in range(1, len(eligibilityTraces)):
            for j in range(len(eligibilityTraces[i])):
                sumE[j]+=eligibilityTraces[i][j]
        
        #calculate weight changes
        deltaW = []
        for i in range(len(self.trainable_weights)):
            deltaW.append(self.learningRate * (float(r) - self.baseline) * sumE[i])
        
        self.optimizer.apply_gradients(zip(deltaW, self.trainable_weights)) #update weights
        
        return {} #TODO what format is this supposed to be in, not really important imo
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch):
        def characteristic_eligibilities(s,a):
            #all wrong, check page 14
            def lng(y, p): #probability mass function, it DOES make sense to calculate this all at once, says so in the paper
                return tfp.distributions.Categorical(p).log_prob(y)
            #based on https://keras.io/api/optimizers/
            
            #first, calculate the log(prob mass value) 
            with tf.GradientTape() as tape:
                output = self(s) #forward pass
                probMass = lng(a, output)

            #then calculate the gradient
            return tape.gradient(probMass, self.trainable_weights)
        #epoch ends, reset env, observation, & reward
        if endOfEpoch:
            #train model
            #zip observations & rewards, pass to fit
            r = rewardsThisEpoch[-1]
            eligibilityTraces = []
            
            for (s,a) in zip(observationsThisEpoch, actionsThisEpoch):
                eligibilityTraces.append(characteristic_eligibilities(s,a))
            
            self.train_step(eligibilityTraces, r)