import unittest
from agents import Agent, RandomAgent, PPOAgent, AdvantageActorCriticAgent, ActorCriticAgent, REINFORCE_MENTAgent, DQNAgent
from environments import TestBanditEnv, MazeEnv, MazeSquare, MazeCoin, TagEnv, TTTEnv, TTTSearchAgent
import tensorflow as tf
from keras import layers
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
    def test_step(self):
        REWARD_PER_COIN = 50
        REWARD_EXPLORATION = 1
        SQUARES = [[MazeSquare.EMPTY] * 5] * 4
        SQUARES.append([MazeSquare.SOLID, MazeSquare.EMPTY, MazeSquare.EMPTY, MazeSquare.EMPTY, MazeSquare.EMPTY])

        env = MazeEnv(nCoins=0, startPosition=(0,0), squares=SQUARES, gameLength=10)
        env.coins.append(MazeCoin((1,0)))
        env.coins.append(MazeCoin((1,1)))

        expectedObservation = [
            [0.0, 4.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ]
        observation, reward, truncated, terminated, _ = env.step(3) #move right
        self.assertFalse(truncated)
        self.assertFalse(terminated)
        self.assertEqual(observation, expectedObservation)
        self.assertEqual(reward, REWARD_EXPLORATION)

        expectedObservation = [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 6.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ]
        observation, reward, truncated, terminated, _ = env.step(2) #move down
        self.assertFalse(truncated)
        self.assertFalse(terminated)
        self.assertEqual(observation, expectedObservation)
        self.assertEqual(reward, REWARD_EXPLORATION + REWARD_PER_COIN)

        expectedObservation = [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [6.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ]
        observation, reward, truncated, terminated, _ = env.step(1) #move left
        self.assertFalse(truncated)
        self.assertFalse(terminated)
        self.assertEqual(observation, expectedObservation)
        self.assertEqual(reward, REWARD_EXPLORATION + REWARD_PER_COIN)

        expectedObservation = [
            [4.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ]
        observation, reward, truncated, terminated, _ = env.step(0) #move up
        self.assertFalse(truncated)
        self.assertFalse(terminated)
        self.assertEqual(observation, expectedObservation)
        self.assertEqual(reward, 0)

        expectedObservation = [
            [4.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ]
        observation, reward, truncated, terminated, _ = env.step(4) #pass
        self.assertFalse(truncated)
        self.assertFalse(terminated)
        self.assertEqual(observation, expectedObservation)
        self.assertEqual(reward, 0)
        
        #check edge collision
        observation, reward, truncated, terminated, _ = env.step(0) #up
        self.assertFalse(truncated)
        self.assertFalse(terminated)
        self.assertEqual(observation, expectedObservation)
        self.assertEqual(reward, 0)
        
        #check wall collision
        expectedObservation = [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ]
        env.step(2) #down
        env.step(2) #down
        env.step(2) #down
        observation, reward, truncated, terminated, _ = env.step(2) #down
        self.assertTrue(truncated)
        self.assertFalse(terminated)
        self.assertEqual(observation, expectedObservation)
        self.assertEqual(reward, 0)
    def test_reset(self):
        SQUARES = [[MazeSquare.EMPTY] * 5] * 4
        SQUARES.append([MazeSquare.SOLID, MazeSquare.EMPTY, MazeSquare.EMPTY, MazeSquare.EMPTY, MazeSquare.EMPTY])
        AGENT_POSITION = (4,4)
        env = MazeEnv(nCoins=0, squares=SQUARES, startPosition=AGENT_POSITION)
        agent = RandomAgent(env.validActions)
        observation = tf.expand_dims(tf.convert_to_tensor(env.reset()[0]), 0)
        #run a random agent for an episode to mess up the state
        running = True
        while running:
            observation, _, terminated, truncated, _ = env.step(agent.act(observation))
            observation = tf.expand_dims(tf.convert_to_tensor(observation), 0)
            running = not (terminated or truncated)
        #call reset & assert that the state has been properly reset
        expectedObservation = [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 4.0],
        ]
        observation, _ = env.reset()
        self.assertEqual(observation, expectedObservation)
        self.assertFalse(env.truncated)
        self.assertFalse(env.terminated)
        self.assertEqual(env.time, 0)
        self.assertEqual(env.coins, [])
        self.assertEqual(env.visited, [AGENT_POSITION])
        self.assertEqual(env.PLAYER_AVATAR.coords, AGENT_POSITION)
    def test_validActions(self):
        env = MazeEnv()
        self.assertEqual(env.validActions(None), [0,1,2,3])
class TestTag(unittest.TestCase):
    def test_logits(self):
        env = TagEnv()
        observation = env.reset()[0]
        xR, yR = env.RUNNER.getCenter()
        rR = env.RUNNER.rotation
        xS, yS = env.SEEKERS[0].getCenter()
        self.assertEqual(observation, [float(xR),float(yR),rR, float(xS),float(yS)])
class TestTTTSearchAgent(unittest.TestCase):
    def test_act(self):
        agent = TTTSearchAgent(random=random.Random(), epsilon=0)
        #0.0 = Empty
        #1.0 = Player
        #2.0 = Enemy
        s = tf.convert_to_tensor([[2,0,0, 0,0,0, 0,0,0]])
        self.assertEqual(agent.act(s), 4)

        s = tf.convert_to_tensor([[0,0,0, 0,2,0, 0,0,0]])
        self.assertEqual(agent.act(s), 0)

        s = tf.convert_to_tensor([[1,1,0, 0,0,0, 0,0,0]])
        self.assertEqual(agent.act(s), 2)

        s = tf.convert_to_tensor([[1,0,0, 0,1,0, 0,0,0]])
        self.assertEqual(agent.act(s), 8)

        s = tf.convert_to_tensor([[2,0,1, 2,0,0, 0,1,0]])
        self.assertEqual(agent.act(s), 6)
class TestTTT(unittest.TestCase):
    def test_step(self):
        REWARD_TIME = 1
        REWARD_PARTIAL_LINE_BASE = 2
        REWARD_COMPLETE_LINE_BASE = 10
        env = TTTEnv(opponent=TTTSearchAgent(random=random.Random(), epsilon=0))
        for i in range(8): #check that all actions work
            observation, reward, truncated, terminated, _ = env.step(i)
            self.assertFalse(truncated)
            self.assertFalse(terminated)
            self.assertEqual(observation[i], 1)
            self.assertEqual(reward, REWARD_TIME + REWARD_PARTIAL_LINE_BASE**1)
            env.reset()
        #0.0 = Empty
        #1.0 = Player
        #2.0 = Enemy
            
        #simulate a partial game to check reward for partial chains
        observation, reward, truncated, terminated, _ = env.step(0)
        self.assertFalse(truncated)
        self.assertFalse(terminated)
        self.assertEqual(observation[0], 1) #our move
        self.assertEqual(observation[4], 2) #only optimal move
        self.assertEqual(reward, REWARD_TIME + REWARD_PARTIAL_LINE_BASE**1)
        
        observation, reward, truncated, terminated, _ = env.step(1)
        self.assertFalse(truncated)
        self.assertFalse(terminated)
        self.assertEqual(observation[0], 1) #our first move
        self.assertEqual(observation[1], 1) #our second move
        #don't check opponent because there are a few optimal moves
        self.assertEqual(reward, REWARD_TIME + REWARD_PARTIAL_LINE_BASE**2)

        #simulate a winning game
        class TestAgent(Agent): #always chooses the action with the lowest index
            def __init__(self) -> None:
                super().__init__()
            def act(self, s):
                for i in range(8):
                    if s[0][i] == 0.0:
                        return i
            def handleStep(self, _, __, ___, ____, _____):
                pass
        env = TTTEnv(opponent=TestAgent())
        agent = TTTSearchAgent(random=random.Random(), epsilon=0)
        observation = tf.expand_dims(tf.convert_to_tensor(env.reset()[0]), 0)
        running = True
        while running:
            observation, reward, terminated, truncated, _ = env.step(agent.act(observation))
            observation = tf.expand_dims(tf.convert_to_tensor(observation), 0)
            running = not (terminated or truncated)
        self.assertTrue(truncated)
        self.assertFalse(terminated)
        #don't check opponent because there are a few optimal moves
        self.assertEqual(reward, REWARD_TIME + REWARD_COMPLETE_LINE_BASE**3)
    def test_reset(self):
        env = TTTEnv()
        agent = RandomAgent(env.validActions)
        observation = tf.expand_dims(tf.convert_to_tensor(env.reset()[0]), 0)
        #run a random agent for an episode to mess up the state
        running = True
        while running:
            observation, _, terminated, truncated, _ = env.step(agent.act(observation))
            observation = tf.expand_dims(tf.convert_to_tensor(observation), 0)
            running = not (terminated or truncated)
        #call reset & assert that the state has been properly reset
        expectedObservation = [0,0,0, 0,0,0, 0,0,0]
        observation, _ = env.reset()
        self.assertEqual(observation, expectedObservation)
        self.assertFalse(env.truncated)
        self.assertFalse(env.terminated)
    def test_validActions(self):
        env = TTTEnv(opponent=TTTSearchAgent(random=random.Random(), epsilon=0))
        observation = [env.reset()[0]]
        self.assertEqual(env.validActions(observation), [0,1,2,3,4,5,6,7,8])
        observation = [env.step(0)[0]]
        self.assertEqual(env.validActions(observation), [1,2,3,5,6,7,8])

if __name__ == '__main__':
    unittest.main()