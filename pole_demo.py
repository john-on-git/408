import gym
import tensorflow as tf
import random
import sys
from pole_train import QPoleModel

if __name__ == "__main__":
    model = QPoleModel(learningRate=0, discountRate=0)
    model.load_weights("checkpoints/QPoleModel_EVO_20231119_183120.tf") #TODO

    env = gym.make('CartPole-v1', render_mode="human")
    observation, _ = env.reset()
    observation = tf.expand_dims(tf.convert_to_tensor(observation),0)
    totalReward=0
    while True:
        #prompt agent
        action = model.act(observation)

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
    env.close()