import tensorflow as tf
import tensorflow.python.keras.models as tfmodels
import tensorflow.python.keras.layers as tflayers
import tensorflow_probability as tfp
import numpy as np

class REINFORCEModel(tfmodels.Model):
    def __init__(self, learningRate, discountRate, baseline=0):
        super().__init__()
        self.learningRate = learningRate
        self.discountRate = discountRate
        self.baseline = baseline
        self.modeltflayers = [
            tflayers.Flatten(input_shape=(1, 4)),
            tflayers.Dense(16, input_shape=(1, 4), activation=tf.nn.relu),
            tflayers.Dense(32, input_shape=(1, 4), activation=tf.nn.relu),
            tflayers.Dense(2, activation=tf.nn.sigmoid),
            tflayers.Softmax()
        ]
        self.compile(
            optimizer="sgd",
            metrics="loss"
        )
    def call(self, observation):
        for layer in self.modeltflayers:
            observation = layer(observation)
        return observation
    def act(self, observation):
        a = int(tf.random.categorical(logits=self(observation),num_samples=1))
        assert a<=2 #categorical can return nCategories+1 (when spread is all zeros?)
        return a
    def train_step(self, eligibilityTraces, datum):
        _, _, r = datum

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
        def discount_and_normalise_rewards(rewards, discountRate=self.discountRate):
            discountedRewards = rewards.copy()
            for i in range(len(discountedRewards)-2, -1, -1): #-2 not a typo, skip the most recent reward
                discountedRewards[i] = rewards[i] * discountRate
                discountRate *= discountRate
            mean = np.mean(discountedRewards)
            sd = np.std(discountedRewards)
            return tf.convert_to_tensor([(discountedReward-mean)/sd for discountedReward in discountedRewards])
        def characteristic_eligibility(datum):
            def lngi(output, action):
                return (tfp.distributions.Categorical(output).log_prob(action))
            #based on https://keras.io/api/optimizers/
            s, a, _ = datum

            #first, we need to calculate the eligibility trace
            with tf.GradientTape() as tape:
                output = self(s) #forward pass
                e = lngi(output, a)

            return tape.gradient(e, self.trainable_weights)
        #epoch ends, reset env, observation, & reward
        if endOfEpoch:
            #train model
            #zip observations & rewards, pass to fit
            discountedRewardsThisEpoch = discount_and_normalise_rewards(rewardsThisEpoch)
            data = zip(observationsThisEpoch, actionsThisEpoch, discountedRewardsThisEpoch)
            
            eligibilityTraces = []
            for datum in data:
                eligibilityTraces.append(characteristic_eligibility(datum))
                self.train_step(eligibilityTraces, datum)

class REINFORCEModel2(tfmodels.Model):
    def __init__(self, learningRate, discountRate, baseline=0):
        super().__init__()
        self.learningRate = learningRate
        self.discountRate = discountRate
        self.baseline = baseline
        self.modeltflayers = [
            tflayers.Flatten(input_shape=(1, 4)),
            tflayers.Dense(16, input_shape=(1, 4), activation=tf.nn.relu),
            tflayers.Dense(32, input_shape=(1, 4), activation=tf.nn.relu),
            tflayers.Dense(2, activation=tf.nn.sigmoid),
            tflayers.Softmax()
        ]
        self.compile(
            optimizer="sgd",
            metrics="loss"
        )
    def call(self, observation):
        for layer in self.modeltflayers:
            observation = layer(observation)
        return observation
    def act(self, observation):
        a = int(tf.random.categorical(logits=self(observation),num_samples=1))
        assert a<=2 #categorical can return nCategories+1 (when spread is all zeros?)
        return a
    def train_step(self, eligibilityTraces, datum):
        _, _, r = datum

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
        def discount_and_normalise_rewards(rewards, discountRate=self.discountRate):
            discountedRewards = rewards.copy()
            for i in range(len(discountedRewards)-2, -1, -1): #-2 not a typo, skip the most recent reward
                discountedRewards[i] = rewards[i] * discountRate
                discountRate *= discountRate
            mean = np.mean(discountedRewards)
            sd = np.std(discountedRewards)
            return tf.convert_to_tensor([(discountedReward-mean)/sd for discountedReward in discountedRewards])
        def characteristic_eligibility(datum):
            def lngi(output, action):
                return tfp.distributions.Categorical(output).log_prob(action)
            #based on https://keras.io/api/optimizers/
            s, a, _ = datum

            with tf.GradientTape() as tape:
                output = self(s) #forward pass
                e = lngi(output, a)

            return tape.gradient(e, self.trainable_weights)
        #epoch ends, reset env, observation, & reward
        if endOfEpoch:
            #train model
            discountedRewardsThisEpoch = discount_and_normalise_rewards(rewardsThisEpoch)
            
            for i in range(len(observationsThisEpoch)):
                eligibilityTraces = []
                for j in range(i+1):
                    eligibilityTraces.append(characteristic_eligibility((observationsThisEpoch[j], actionsThisEpoch[j], discountedRewardsThisEpoch[j])))
                self.train_step(eligibilityTraces, (observationsThisEpoch[i], actionsThisEpoch[i], discountedRewardsThisEpoch[i]))