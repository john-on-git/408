import sys
import datetime
import random
import gym
import numpy as np
import tensorflow as tf
import tensorflow as tfp
import tensorflow.python.keras.models as models
import tensorflow.python.keras.layers as layers
import matplotlib.pyplot as plt

BASELINE = 0

#layers.Dense(4, input_shape=(1, 4), activation=tf.nn.relu),
#layers.Dense(8, input_shape=(1, 4), activation=tf.nn.relu),
#layers.Dense(16, input_shape=(1, 4), activation=tf.nn.relu),
#layers.Dense(32, input_shape=(1, 4), activation=tf.nn.relu),
#layers.Dense(32, input_shape=(1, 4), activation=tf.nn.relu),
#layers.Dense(32, input_shape=(1, 4), activation=tf.nn.relu),
#layers.Dense(32, input_shape=(1, 4), activation=tf.nn.relu),
#layers.Dense(32, input_shape=(1, 4), activation=tf.nn.relu),
#layers.Dense(32, input_shape=(1, 4), activation=tf.nn.relu),   
#layers.Dense(2, activation=tf.nn.sigmoid)

class QPoleModel(models.Model):
    def __init__(self, learningRate, discountRate):
        super().__init__()
        self.learningRate = learningRate
        self.discountRate = discountRate
        self.modelLayers = [
            layers.Dense(32, input_shape=(1, 4), activation=tf.nn.relu),
            layers.Dense(64, input_shape=(1, 4), activation=tf.nn.relu),
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
        return int(tf.argmax(tf.reshape(self(observation), shape=(2,1))))
    def train_step(self, datum):
        def f(a, r, s):
            return ((1-self.learningRate) * a) + (self.learningRate * (r + self.discountRate * tf.reduce_max(self(s))))
        #based on https://keras.io/api/optimizers/
        s1, r, s2 = datum
        
        with tf.GradientTape() as tape:
            a = self(s1) #forward pass
            valueToOptimiseFor = f(a, r, s2)

        gradients = tape.gradient(valueToOptimiseFor, self.trainable_weights)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights)) #update weights
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch):
        if len(observationsThisEpoch)>1:
            self.train_step((observationsThisEpoch[-2], rewardsThisEpoch[-2], observationsThisEpoch[-1]))

class REINFORCEPoleModel(models.Model):
    def __init__(self, learningRate, discountRate):
        super().__init__()
        self.learningRate = learningRate
        self.discountRate = discountRate
        self.modelLayers = [
            layers.Dense(4, input_shape=(1, 4), activation=tf.nn.relu),
            layers.Dense(8, input_shape=(1, 4), activation=tf.nn.relu),
            layers.Dense(16, input_shape=(1, 4), activation=tf.nn.relu),
            layers.Dense(32, input_shape=(1, 4), activation=tf.nn.relu),
            layers.Dense(2, activation=tf.nn.sigmoid),
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
        return int(tf.random.categorical(logits=self(observation),num_samples=1))
    def train_step(self, datum):
        def ln_gᵢ(output, action):
            m = tfp.distributions.Categorical(output)
            return (m.log_prob(action))
        #based on https://keras.io/api/optimizers/
        observation, action, reward = datum
        
        with tf.GradientTape() as tape:
            output = self(observation) #forward pass
            valueToOptimiseFor = ln_gᵢ(output, action)

        gradients = tape.gradient(valueToOptimiseFor, self.trainable_weights)

        #apply reward/baseline/learning rate
        for i in range(len(gradients)):
            gradients[i] = self.learningRate * float(reward - BASELINE) * gradients[i]

        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights)) #update weights
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
            
            #model.fit(data) #commented out bc TF throws errors (why?)
            
            #manual training loop (see above)
            for datum in data:
                model.train_step(datum)

if __name__ == "__main__":
    RNG_SEED=44
    TARGET_EPOCHS_INIT = 500
    random.seed(RNG_SEED)
    tf.random.set_seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    poleModels = [QPoleModel(learningRate=.9, discountRate=.99)]
    for model in poleModels:
        targetEpochs = TARGET_EPOCHS_INIT
        env = gym.make('CartPole-v1')
        env.action_space.seed(RNG_SEED)
        
        observation, info = env.reset(seed=RNG_SEED)
        observation = tf.expand_dims(tf.convert_to_tensor(observation),0)
        
        rewardsOverall = []
        epochs = 0
        print("Training new ", type(model).__name__)
        while True:
            while epochs < targetEpochs:
                Ss = []
                As = []
                Rs = []
                t = 0
                stop = False
                while not stop:
                    Ss.append(observation) #record observation for training

                    #prompt agent
                    action = model.act(tf.convert_to_tensor(observation))
                    As.append(action)

                    #pass action to env, get next observation
                    nextObservation, reward, terminated, truncated, _ = env.step(action)
                    Rs.append(reward)
                    
                    nextObservation = tf.convert_to_tensor(nextObservation)
                    nextObservation = tf.expand_dims(nextObservation,0)

                    model.handleStep(terminated or truncated, Ss, As, Rs) #template method pattern 🤓
                    
                    if terminated or truncated:
                        nextObservation, _ = env.reset(seed=RNG_SEED)
                        nextObservation = tf.convert_to_tensor(nextObservation)
                        nextObservation = tf.expand_dims(nextObservation,0)
                        
                        #calc overall reward for graph
                        rewardsOverall.append(sum(Rs))
                        Ss.clear()
                        As.clear()
                        Rs.clear()
                        stop = True
                    observation = nextObservation
                    t+=1
                epochs+=1
                print("Epoch ", epochs, " Done (duration ", t, ")", sep="")

            n = input("enter number to extend training, non-numeric to end\n")
            if(n.isnumeric()):
                targetEpochs+=int(n)
            else:
                model.save_weights("checkpoints\\" + type(model).__name__ + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".tf", overwrite=True, save_format="tf")
                break
        env.close()
            
        #plot return over time
        x = range(len(rewardsOverall))
        y = rewardsOverall
        plt.scatter(x,y, label=type(model).__name__)
        m, c = np.polyfit(x,y,1)
        plt.plot(m*x + c) #line of best fit
    plt.legend()
    plt.show()
    exit()
    while True:
        #prompt agent
        output = model.call(tf.convert_to_tensor(observation))
        action = int(tf.random.categorical(logits=output,num_samples=1))

        #pass action to env, get next observation
        observation, reward, terminated, truncated, info = env.step(action)
        observation = tf.expand_dims(tf.convert_to_tensor(observation),0)
        totalReward+=reward
        #epoch ends, reset env, observation, & reward
        if terminated or truncated:
            print("Reward:", totalReward)
            totalReward=0
            observation, info = env.reset(seed=random.randint(0,sys.maxsize))
            observation = tf.expand_dims(tf.convert_to_tensor(observation),0)