import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from maze_env import MazeEnv
from jgw_cs408.agents import *
import time

if __name__ == "__main__":
    RNG_SEED_INIT=42
    TRAINING_TIME_SECONDS = 600

    agents = [
        #REINFORCEAgent(learningRate=.01, discountRate=.75, baseline=0),
        #MonteCarloAgent(learningRate=.001, discountRate=.75, replayMemoryCapacity=5000, replayFraction=25, epsilon=.99, epsilonDecay=.9999),
        #SARSAAgent(learningRate=.001, discountRate=.95, replayMemoryCapacity=1000, epsilon=.75, epsilonDecay=.999),
        DQNAgent(learningRate=.001, discountRate=.75, replayMemoryCapacity=5000, replayFraction=25, epsilon=.99, epsilonDecay=.999),
        #ActorCriticAgent(learningRate=.001, discountRate=.75, replayMemoryCapacity=50000, replayFraction=500, epsilon=.99, epsilonDecay=.9999),
        #REINFORCEAgent(learningRate=.01, discountRate=.75, baseline=0),
        #REINFORCE_MENTAgent(learningRate=.01, discountRate=.75, baseline=0)
    ]

    metrics = {"reward":[], "loss":[]}
    for i in range(len(agents)):
        metrics["reward"].append(list())
        metrics["loss"].append(list())
    random.seed(RNG_SEED_INIT)
    tf.random.set_seed(RNG_SEED_INIT)
    np.random.seed(RNG_SEED_INIT)

    for i in range(len(agents)):
        print("Training new ", type(agents[i]).__name__)
        epochs = 0
        env = MazeEnv(nCoins=10, startPosition="random")
        rngSeed=RNG_SEED_INIT
        observation, _ = env.reset(seed=rngSeed)
        observation = tf.expand_dims(tf.convert_to_tensor(observation),0)
        start = time.time()
        while time.time()-start<=TRAINING_TIME_SECONDS:
            Ss = []
            As = []
            Rs = []
            Losses = []
            epochRunning = True
            while epochRunning: #for each time step in epoch
                Ss.append(observation) #record observation for training

                #prompt agent
                action = agents[i].act(tf.expand_dims(tf.convert_to_tensor(observation),0))
                As.append(action) #record observation for training

                #pass action to env, get next observation
                observation, reward, terminated, truncated, _ = env.step(action)
                Rs.append(float(reward)) #record observation for training
                observation = tf.expand_dims(tf.convert_to_tensor(observation),0)

                agents[i].handleStep(terminated or truncated, Ss, As, Rs, callbacks=[
                    tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: Losses.append(logs["loss"]))
                ])
                
                if terminated or truncated:
                    observation, _ = env.reset(seed=rngSeed)
                    observation = tf.expand_dims(tf.convert_to_tensor(observation),0)
                    
                    metrics["reward"][i].append(sum(Rs)) #calc overall reward for graph
                    metrics["loss"][i].extend(Losses) #calc overall reward for graph
                    Ss = []
                    As = []
                    Rs = []
                    epochRunning = False
                    print("Epoch ", epochs, " Done (r = ", metrics["reward"][i][-1],", ε ≈ ", round(agents[i].epsilon, 2),")", sep="")
                observation = observation
            epochs+=1
            rngSeed+=1
        #finished training this agent
    #finished training all agents
            
    #save the agents
    for agent in agents:
        #path = "checkpoints\\" + type(agent).__name__ + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".tf"
        path = "checkpoints\\Maze" + type(agent).__name__ + ".tf"
        agent.save_weights(path, overwrite=True)

    #create reward baseline
    agents.append(RandomAgent())
    metrics["reward"].append(list())
    metrics["loss"].append(list())
    maxLen = 0
    for xs in metrics["reward"]:
        maxLen = max(maxLen, len(xs))
    env = MazeEnv(nCoins=10, startPosition="random")
    epochs = 0
    while epochs<maxLen:
        epochRunning = True
        Rs = []
        observation, _ = env.reset(seed=rngSeed)
        observation = tf.expand_dims(tf.convert_to_tensor(observation),0)
        while epochRunning: #for each time step in epoch
            #prompt agent
            action = agents[-1].act(tf.expand_dims(tf.convert_to_tensor(observation),0))

            #pass action to env, get next observation
            observation, reward, terminated, truncated, _ = env.step(action)
            Rs.append(float(reward)) #record observation for training
            observation = tf.expand_dims(tf.convert_to_tensor(observation),0)
            
            if terminated or truncated:
                metrics["reward"][-1].append(sum(Rs)) #calc overall reward for graph
                epochRunning = False
                epochs+=1
    
    def plot(target, agents, yss, metricName):
        for i in range(len(agents)):
            ys = yss[i]
            x = range(len(ys))
            
            #smooth the curve
            smoothedYs = []
            window = []
            windowSize = 5#max(len(ys)/200, 1)
            for y in ys:
                window.append(y)
                if len(window)>windowSize:
                    window.pop(0)
                smoothedYs.append(sum(window)/windowSize)
            target.plot(x,smoothedYs, label=metricName + "(" + type(agents[i]).__name__ + ")")
            #m, c = np.polyfit(x,ys,1)
            #plt.plot(m*x + c) #line of best fit
        target.legend()

    #plot return over time for each agent
    if not plt.get_fignums() == []:
        plt.clf() #clear previous graph
    fig, axs = plt.subplots(len(metrics))
    i = 0
    for k in metrics:
        plot(axs[i], agents, metrics[k], k)
        i+=1
    plt.show()
    
    #prompt to continue training
    #n = input("enter number to extend training, non-numeric to end\n")
    #if(n.isnumeric()):
    #    resetEpochs=epochs
    #    targetEpochs+=int(n)
    #else:
    #    trainingRunning = False
    #    env.close()
    input("press any key to continue")
    exit()