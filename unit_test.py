import unittest
from agents import Agent, PPOAgent, AdvantageActorCriticAgent, ActorCriticAgent, REINFORCE_MENTAgent, DQNAgent
from environments import TestBanditEnv, MazeEnv, TagEnv, TTTEnv
import tensorflow as tf
from keras import layers
import os
import random
import numpy as np

#this is the closest thing to a unit test I could come up with
class TestAgents(unittest.TestCase):
    def template_test_agent(self, agent):
        random.seed(1)
        tf.random.set_seed(1)
        np.random.seed(1)
        environment = TestBanditEnv()
        agent = agent
        rewardPerEpisode = []
        N_EPISODES = 200
        for _ in range(N_EPISODES):
            Ss = []
            As = []
            Rs = []
            observation, _ = environment.reset()
            observation = tf.expand_dims(tf.convert_to_tensor(observation), 0)
            Ss.append(observation) #record observation for training
            terminated = False
            truncated = False
            while not (terminated or truncated): #for each time step in epoch

                #prompt agent
                action = agent.act(tf.convert_to_tensor(observation))
                As.append(action) #record action for training

                #pass action to environment, get next observation
                observation, reward, terminated, truncated, _ = environment.step(action)
                observation = tf.expand_dims(tf.convert_to_tensor(observation), 0)
                Rs.append(float(reward)) #record reward for training
                Ss.append(observation) #record observation for training

                agent.handleStep(terminated or truncated, Ss, As, Rs)
            rewardPerEpisode.append(sum(Rs))
        average = sum(rewardPerEpisode[-10:])/10
        self.assertGreater(average, .95)
    def test_PPO(self):
        agent = PPOAgent(
            actionSpace=[0,1],
            hiddenLayers=[layers.Flatten(), layers.Dense(4, activation=tf.nn.sigmoid)],
            validActions=lambda s: [0,1],
            learningRate=.01,
            epsilon=.5,
            epsilonDecay=.66
        )
        self.template_test_agent(agent)
    def test_A2C(self):
        agent = AdvantageActorCriticAgent(
            actionSpace=[0,1],
            hiddenLayers=[layers.Flatten(), layers.Dense(4, activation=tf.nn.sigmoid)],
            validActions=lambda s: [0,1],
            learningRate=.01,
            epsilon=.5,
            epsilonDecay=.66,
            criticWeight=.1
        )
        self.template_test_agent(agent)
    def test_ActorCritic(self):
        agent = ActorCriticAgent(
            actionSpace=[0,1],
            hiddenLayers=[layers.Flatten(), layers.Dense(4, activation=tf.nn.sigmoid)],
            validActions=lambda s: [0,1],
            learningRate=.01,
            epsilon=.5,
            epsilonDecay=.66,
            replayMemoryCapacity=200,
            replayFraction=20
        )
        self.template_test_agent(agent)
    def test_REINFORCE(self):
        agent = REINFORCE_MENTAgent(
            actionSpace=[0,1],
            hiddenLayers=[layers.Flatten(), layers.Dense(4, activation=tf.nn.sigmoid)],
            validActions=lambda s: [0,1],
            learningRate=.01,
            epsilon=.5,
            epsilonDecay=.66
        )
        self.template_test_agent(agent)
    def test_DQN(self):
        agent = DQNAgent(
            actionSpace=[0,1],
            hiddenLayers=[layers.Flatten(), layers.Dense(4, activation=tf.nn.sigmoid)],
            validActions=lambda s: [0,1],
            learningRate=.01,
            epsilon=.5,
            epsilonDecay=.66,
            replayMemoryCapacity=200,
            replayFraction=20
        )
        self.template_test_agent(agent)

class TestMaze(unittest.TestCase):
    def test_logits(self):
        env = MazeEnv()
        env.SQUARES
class TestTTTSearchAgent():
    pass
class TestTag(unittest.TestCase):
    def test_name(self):
        pass
class TestTTT(unittest.TestCase):
    pass

if __name__ == '__main__':
    unittest.main()