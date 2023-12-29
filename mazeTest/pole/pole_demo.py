import gym
import tensorflow as tf
from pole_agents import *

if __name__ == "__main__":
    model = SARSAAgent(learningRate=.75, discountRate=.75, replayMemoryCapacity=1000)
    model.load_weights("checkpoints\PoleSARSAAgent.tf") #TODO

    env = gym.make('CartPole-v1', render_mode="human")
    env.action_space.seed()

    observation, _ = env.reset()
    observation = tf.expand_dims(tf.convert_to_tensor(observation),0)

    totalReward=0
    while True:
        #prompt agent
        action = model.act(tf.convert_to_tensor(observation))

        #pass action to env, get next observation
        observation, reward, terminated, truncated, _ = env.step(action)
        totalReward+=reward
        observation = tf.expand_dims(tf.convert_to_tensor(observation),0)
        
        if terminated or truncated:
            observation, _ = env.reset()
            observation = tf.expand_dims(tf.convert_to_tensor(observation),0)
            
            #calc overall reward for graph
            print("reward:", totalReward)
            totalReward = 0