import tensorflow as tf
import tensorflow.python.keras.models as tfmodels
import tensorflow.python.keras.layers as tflayers

class QLearningModel(tfmodels.Model):
    def __init__(self, learningRate, discountRate):
        super().__init__()
        self.learningRate = learningRate
        self.discountRate = discountRate
        self.modelLayers = [
            #tflayers.ZeroPadding2D(padding=(100,100)),
            tflayers.Flatten(input_shape=(1, 5, 5)),
            tflayers.Dense(32, activation=tf.nn.relu),
            tflayers.Dense(16, activation=tf.nn.relu),
            tflayers.Dense(5, activation=tf.nn.sigmoid)
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
        return int(tf.argmax(tf.reshape(self(observation), shape=(5,1))))
    def train_step(self, datum):
        def f(a, r, s):
            return ((1-self.learningRate) * a) + (self.learningRate * (r + self.discountRate * tf.reduce_max(self(s))))
        #based on https://keras.io/api/optimizers/
        s1, r, s2 = datum

        a = self(s1) #forward pass
        self.optimizer.minimize(lambda: f(a,r,s2), self.trainable_weights)
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch):
        if len(observationsThisEpoch)>1:
            self.train_step((observationsThisEpoch[-2], rewardsThisEpoch[-2], observationsThisEpoch[-1]))