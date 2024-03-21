import random
import numpy as np
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
from environments import Environment, MazeEnv, TagEnv, TTTEnv
from agents import *
import time
import os

if __name__ == "__main__":
    RNG_SEED_INIT = 0 #fixed RNG for replicability.
    N_EPISODES = 1000 #number of episodes to train for
    N_AGENTS = 5 #used to init metrics. Must be equal to len(agents). There's probably a better way to do this but I don't want to overcomplicate it.

    environments: list[Environment]
    environments = [
        MazeEnv(),
        #TTTEnv(),
        #TagEnv(),
    ]
    metrics = np.ndarray(shape=(len(environments), N_AGENTS, N_EPISODES, 2))
    for i in range(len(environments)):
        agents: list[Agent]
        agents = [
            DQNAgent(
               actionSpace=environments[i].actionSpace,
               hiddenLayers=[layers.Flatten(), layers.Dense(16, activation=tf.nn.sigmoid)],
               validActions=environments[i].validActions,
               learningRate=.001,
               discountRate=.9,
               epsilon=.5,
               epsilonDecay=.99,
               replayMemoryCapacity=1000,
               replayFraction=10
            ),
            AdvantageActorCriticAgent(
                actionSpace=environments[i].actionSpace,
                hiddenLayers=[layers.Flatten(), layers.Dense(16, activation=tf.nn.sigmoid)],
                validActions=environments[i].validActions,
                learningRate=.001,
                discountRate=.9,
                epsilon=.25,
                epsilonDecay=.99,
                criticWeight=10
            ),
            PPOAgent(
                actionSpace=environments[i].actionSpace,
                hiddenLayers=[layers.Flatten(), layers.Dense(16, activation=tf.nn.sigmoid)],
                validActions=environments[i].validActions,
                learningRate=.001,
                discountRate=.9,
                epsilon=.25,
                epsilonDecay=.99,
            ),
            ActorCriticAgent(
                actionSpace=environments[i].actionSpace,
                hiddenLayers=[layers.Flatten(), layers.Dense(16, activation=tf.nn.sigmoid)],
                validActions=environments[i].validActions,
                learningRate=.001,
                discountRate=.9,
                epsilon=.25,
                epsilonDecay=.99,
                replayFraction=10
            ),

            REINFORCE_MENTAgent(
                actionSpace=environments[i].actionSpace,
                hiddenLayers=[layers.Flatten(), layers.Dense(16, activation=tf.nn.sigmoid)],
                validActions=environments[i].validActions,
                learningRate=.001,
                discountRate=.9,
                epsilon=.5,
                epsilonDecay=.99
            ),
        ]
        assert len(agents) == N_AGENTS
        for j in range(len(agents)):
            print("Training new", type(environments[i]).__name__,"/",type(agents[j]).__name__)

            rngSeed=RNG_SEED_INIT
            start = time.time()
            for k in range(N_EPISODES):
                random.seed(rngSeed)
                tf.random.set_seed(rngSeed)
                np.random.seed(rngSeed)
                Ss = []
                As = []
                Rs = []
                #Losses = []
                observation, _ = environments[i].reset(rngSeed)
                observation = tf.expand_dims(tf.convert_to_tensor(observation), 0)
                Ss.append(observation) #record observation for training
                terminated = False
                truncated = False
                while not (terminated or truncated): #for each time step in episode

                    #prompt agent
                    action = agents[j].act(tf.convert_to_tensor(observation))
                    As.append(action) #record action for training

                    #pass action to environment, get next observation
                    observation, reward, terminated, truncated, _ = environments[i].step(action)
                    observation = tf.expand_dims(tf.convert_to_tensor(observation), 0)
                    Rs.append(float(reward)) #record reward for training
                    Ss.append(observation) #record observation for training
                    


                    agents[j].handleStep(terminated or truncated, Ss, As, Rs, callbacks=[
                        #tf.keras.callbacks.LambdaCallback(on_episode_end=lambda _, logs: Losses.append(logs["loss"])) #for logging loss as a metric (not used atm)
                    ])
                #episode finished
                metrics[i][j][k][0] = sum(Rs)
                metrics[i][j][k][1] = time.time()
                print("Episode ", k, " Done (r = ", metrics[i][j][k][0],", ε = ", round(agents[j].epsilon,2), ")", sep="")
                rngSeed+=1
            #finished training this agent
            #write the model weights to file
            weightsPath = "checkpoints\\" + type(environments[i]).__name__ + "_" + type(agents[j]).__name__ + ".tf"
            agents[j].save_weights(weightsPath, overwrite=True)
        #finished training all agents on this environment
    #finished training all environments
    #write the metrics to file
    metricsDir = os.path.dirname(os.path.abspath(__file__)) + "\\metrics"
    os.makedirs(metricsDir, exist_ok=True)
    metadataAgents = [type(x).__name__ for x in agents]
    metadataEnvironments = [type(x).__name__ for x in environments]
    np.savez(
        metricsDir + "\\metrics_" + time.strftime("%Y.%m.%d-%H.%M.%S"),
        agents=metadataAgents,
        environments=metadataEnvironments,
        nEpisodes = [N_EPISODES],
        data=metrics
    )

    def plot(target, yss, i,m, label):
        match type(environments[i]).__name__:
            case "MazeEnv":
                target.axhline(y=571/2, color="grey")
            case "TagEnv":
                target.axhline(y=565/2, color="grey")
            case "TTTEnv":
                target.axhline(y=1000/2, color="lightgrey")
        for j in range(len(agents)):
            ys = []
            for k in range(N_EPISODES):
                ys.append(yss[i][j][k][m])

            x = range(len(ys))

            #smooth the curve
            smoothedYs = []
            window = []
            windowSize = 5 #max(len(ys)/200, 1)
            for y in ys:
                window.append(y)
                if len(window)>windowSize:
                    window.pop(0)
                smoothedYs.append(sum(window)/windowSize)
            target.plot(x,smoothedYs, label=label + "(" + type(agents[j]).__name__ + ")")
            #m, c = np.polyfit(x,ys,1)
            #plt.plot(m*x + c) #line of best fit
            target.title.set_text(type(environments[i]).__name__)
            target.legend()

    #plot return over time for envs/agents
    fig, axs = plt.subplots(nrows=len(environments))
    #print
    if len(environments) == 1: #axs is a list, unless the first dimension would be length 1, then it isn't. account for that.
        plot(axs, metrics, 0,0, "reward")
    else:
        for i in range(len(environments)):
            plot(axs[i], metrics, i,0, "reward")
    plt.show()

    input("press any key to continue") #because 6pyplot fails to show sometimes
    #prompt to continue training
    
    #n = input("enter number to extend training, non-numeric to end\n")
    #if(n.isnumeric()):
    #    resetEpisodes=episodes
    #    targetEpisodes+=int(n)
    #else:
    #    trainingRunning = False
    #    environments[i].close()