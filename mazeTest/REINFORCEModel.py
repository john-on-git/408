import numpy as np
import tensorflow as tf
import tensorflow.python.keras.models as tfmodels
import tensorflow.python.keras.layers as tflayers
import tensorflow_probability as tfp

class REINFORCEModel(tfmodels.Model):
    def __init__(self, learningRate, discountRate):
        super().__init__()
        self.learningRate = learningRate
        self.discountRate = discountRate
        self.modeltflayers = [
            tflayers.Flatten(input_shape=(1, 5, 5)),
            tflayers.Dense(4, input_shape=(1, 4), activation=tf.nn.relu),
            tflayers.Dense(8, input_shape=(1, 4), activation=tf.nn.relu),
            tflayers.Dense(16, input_shape=(1, 4), activation=tf.nn.relu),
            tflayers.Dense(32, input_shape=(1, 4), activation=tf.nn.relu),
            tflayers.Dense(5, activation=tf.nn.sigmoid),
            tflayers.Softmax()
        ]
        self.compile(
            optimizer="Adam",
            metrics="loss"
        )
    def call(self, observation):
        for layer in self.modeltflayers:
            observation = layer(observation)
        return observation
    def act(self, observation):
        a = int(tf.random.categorical(logits=self(observation),num_samples=1))
        return a if a<=4 else 4 #categorical can return nCategories+1 (when spread is all zeros?)
    def train_step(self, datum):
        def ln_gᵢ(output, action):
            m = tfp.distributions.Categorical(output)
            return (m.log_prob(action))
        #based on https://keras.io/api/optimizers/
        observation, action, reward = datum
        
        self.optimizer.minimize()

        with tf.GradientTape() as tape:
            output = self(observation) #forward pass
            loss = ln_gᵢ(output, action)

        gradients = tape.gradient(loss, self.trainable_weights)

        #apply reward/baseline/learning rate
        for i in range(len(gradients)):
            gradients[i] = self.learningRate * float(reward) * gradients[i]
        return {}
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch):
        def discount_and_normalise_rewards(rewards, discountRate=self.discountRate):
            discountedRewards = rewards.copy()
            for i in range(len(discountedRewards)-2, -1, -1): #-2 not a typo, skip the most recent reward
                discountedRewards[i] = rewards[i] * discountRate
                discountRate *= discountRate
            mean = np.mean(discountedRewards)
            sd = np.std(discountedRewards)
            return tf.convert_to_tensor([(discountedReward-mean)/sd for discountedReward in discountedRewards])
        
        #epoch ends, reset env, observation, & reward
        if endOfEpoch:
            #train model
            #zip observations & rewards, pass to fit
            discountedRewardsThisEpoch = discount_and_normalise_rewards(rewardsThisEpoch)
            data = zip(observationsThisEpoch, actionsThisEpoch, discountedRewardsThisEpoch)
            
            self.fit(data)