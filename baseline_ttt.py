from environments import TTTEnv, TTTSearchAgent
import tensorflow as tf
from random import Random

#this code runs the optimal Tic-Tac-Toe agent to determine its average reward.
#the opponent incudes a random element, so it's kind of hard to predict formally, much easier to take measurements.

rngSeed = 0
env = TTTEnv()
agent = TTTSearchAgent(Random(rngSeed), epsilon=0)

terminated = False
truncated = False
rewardOverall = 0
rewardThisEpisode = 0
running = True
currentEpisode=0
terminated = False
truncated = False
N_EPISODES = 100
while running and currentEpisode<N_EPISODES:
    rewardThisEpisode = 0
    observation = tf.expand_dims(tf.convert_to_tensor(env.reset()[0]),0)
    rngSeed+=1
    while running and not (terminated or truncated):
        observation, reward, terminated, truncated, _ = env.step(agent.act(observation))
        rewardThisEpisode+=reward
        observation = tf.expand_dims(tf.convert_to_tensor(observation),0)
    print(f"reward (episode {currentEpisode+1}):", rewardThisEpisode)
    currentEpisode+=1
    rewardOverall+=rewardThisEpisode
    terminated = False
    truncated = False
    observation, _ = env.reset(rngSeed)
print("num episodes:", N_EPISODES)
print("average reward:", rewardOverall/N_EPISODES)
