import tensorflow as tf
import tensorflow_probability as tfp
from keras import Model
from keras import layers as layers
import numpy as np
import random

class RandomAgent():
    def __init__(self):
        self.epsilon = 1
        pass
    def act(self, s):
        valid = []
        for i in range(9):
            if s[0][i] == 0.0:
                valid.append(i)
        return random.choice(valid)
    def handleStep(self, _, __, ___, ____):
        pass
    def save_weights(self, path, overwrite):
        pass

#agent that follows the optimal policy by performing a full search
class SearchAgent():
    def  __init__(self, epsilon=0, epsilonDecay=0) -> None:
        self.epsilon=epsilon
        self.epsilonDecay = epsilonDecay
    def act(self, s) -> (0|1|2|3|4|5|6|7|8):
        #0.0 = Empty
        #1.0 = Player
        #2.0 = Enemy
        def actionSpace(s):
            actions = []
            for i in range(9):
                if s[0][i]==0:
                    actions.append(i)
            return actions
        def succ(s, a, actor):
            s = s.copy()
            s[0][a] = actor
            return s
        def isTerminal(s) -> (int|None):
            def win(s, actor):
                a = s[0][0]==actor
                b = s[0][1]==actor
                c = s[0][2]==actor
                d = s[0][3]==actor
                e = s[0][4]==actor
                f = s[0][5]==actor
                g = s[0][6]==actor
                h = s[0][7]==actor
                i = s[0][8]==actor
                return (
                        #cardinals
                        (a and ((b and c) or (d and g))) or
                        (e and ((d and f) or (b and h))) or 
                        (i and ((h and g) or (c and f)))
                    ) or (
                        #diagonals
                        e and (
                            (a and i) or
                            (b and h) or
                            (d and f) or
                            (g and c)
                        )
                    )
            def draw(s):
                for x in s[0]:
                    if x==0:
                        return False
                return True
            if win(s,1):
                return 999
            elif win(s,2):
                return -999
            elif draw(s):
                return 0
            else:
                return None
        def mini(s) -> int:
            tV = isTerminal(s)
            if tV==None: #not terminal
                worst = None
                for action in actionSpace(s):
                    score = max(succ(s,action, 2.0))
                    if score==-999:
                        return score
                    if worst==None or score<worst:
                        worst = score
                return worst
            else:
                return tV
        def max(s) -> int:
            tV = isTerminal(s)
            if tV==None:
                best = None
                for action in actionSpace(s):
                    score = mini(succ(s,action, 1.0))
                    if score==999:
                        return score
                    if best==None or score>best:
                        best = score
                return best
            else:
                return tV
        
        self.epsilon *= self.epsilonDecay
        if random.random()<self.epsilon: #chance to act randomly
            valid = []
            for i in range(9):
                if s[0][i] == 0.0:
                    valid.append(i)
            return random.choice(valid)
        else:
            s = s.numpy()
            best = None
            for action in actionSpace(s):
                score = mini(succ(s,action, 1.0))
                if best==None or score>best[1]:
                    best = (action, score)
            return None if best==None else best[0]

class REINFORCE_MENTAgent(Model):
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
            layers.Dense(9, tf.nn.softmax)
        ]
        self.compile(
            optimizer=tf.optimizers.Adam(),
            metrics="loss"
        )
    def call(self, observation):
        for layer in self.modelLayers:
            observation = layer(observation)
        return observation
    def act(self, s):
        self.epsilon*=self.epsilonDecay
        if random.random()<self.epsilon: #chance to act randomly
            valid = []
            for i in range(9):
                if s[0][i] == 0.0:
                    valid.append(i)
            return random.choice(valid)
        else:
            return int(tf.random.categorical(logits=self(s),num_samples=1))
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
            layers.Dense(9, tf.nn.softmax)
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
        valid = []
        for i in range(9):
            if s[0][i] == 0.0:
                valid.append(i)
        if random.random()<self.epsilon: #chance to act randomly
            return random.choice(valid)
        else:
            x = self(s)[0]
            y = np.array([-999,-999,-999,-999,-999,-999,-999,-999,-999])
            for i in range(9):
                if i in valid:
                    y[i] = x[i]
            return int(np.argmax(y)) #otherwise greedy, but only for valid moves
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

class ParallelREINFORCE_MENTAgent(Model):
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
            layers.Dense(9, tf.nn.softmax)
        ]
        self.compile(
            optimizer=tf.optimizers.Adam(),
            metrics="loss"
        )
    def call(self, observation):
        for layer in self.modelLayers:
            observation = layer(observation)
        return observation
    def act(self, s):
        self.epsilon*=self.epsilonDecay
        if random.random()<self.epsilon: #chance to act randomly
            valid = []
            for i in range(9):
                if s[0][i] == 0.0:
                    valid.append(i)
            return random.choice(valid)
        else:
            return int(tf.random.categorical(logits=self(s),num_samples=1))
    def train_step(self, eligibilityTraces, r, gradsCollector):
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
        
        for i in range(len(grads)): #collect grads for async
            if gradsCollector[i]==None:
                gradsCollector[i] = grads[i]
            else:
                gradsCollector[i]+=grads[i]

        self.optimizer.apply_gradients(zip(
            grads,
            self.trainable_weights
        )) #update weights
        return {"loss": (self.baseline-r)}
    def handleStep(self, endOfEpoch, observationsThisEpoch, actionsThisEpoch, rewardsThisEpoch, gradsCollector, callbacks=[]):
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
            
            self.train_step(eligibilityTraces, float(sum(rewardsThisEpoch)) + averageEntropy(observationsThisEpoch), gradsCollector)