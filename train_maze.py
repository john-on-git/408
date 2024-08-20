import random
import numpy as np
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
from environments import MazeEnv
from agents import *
import datetime
import os
import multiprocessing as mp

RNG_SEED = 0 #fixed RNG for replicability.
N_EPISODES = 10000 #number of episodes to train for
N_METRICS = 1 #reward

#it's possible to make these three files one file, but the extra axis would make it really confusing, and this way is more convenient to run

#the parallelism is poorly implemented, but afaik there's no way to pass tf models across threads or processes, since they aren't pickleable.
#This is the best I could come up with.
#environment is defined in this function, agents are defined in main
def train(agentType, agentConfig, metrics, anyProcessFailed):
    try:
        rngSeed=RNG_SEED
        random.seed(rngSeed)
        tf.random.set_seed(rngSeed)
        np.random.seed(rngSeed)
        
        environment = MazeEnv()
        
        #in order for the child process to pass them to the agent constructor, all args must be specified, even if they're unused by this agent
        hiddenLayers, learningRate, epsilon, epsilonDecay, discountRate, entropyWeight, criticWeight, tMax, interval, replayMemoryCapacity, replayFraction = agentConfig
        agent: Agent
        agent = agentType(
            learningRate=learningRate,
            actionSpace=environment.actionSpace,
            hiddenLayers=hiddenLayers,
            validActions=environment.validActions,
            epsilon=epsilon,
            epsilonDecay=epsilonDecay,
            discountRate=discountRate,
            entropyWeight=entropyWeight,
            criticWeight=criticWeight,
            tMax=tMax,
            interval=interval,
            replayMemoryCapacity=replayMemoryCapacity,
            replayFraction=replayFraction
        )
        print("Training new " + type(environment).__name__ + " / " + type(agent).__name__)
        Ss = []
        As = []
        Rs = []
        for i in range(N_EPISODES):
            #Losses = []
            observation, _ = environment.reset(rngSeed)
            observation = tf.expand_dims(tf.convert_to_tensor(observation), 0)
            Ss.append(observation) #record observation for training
            terminated = False
            truncated = False
            while not (terminated or truncated): #for each time step in episode

                #prompt agent
                action = agent.act(tf.convert_to_tensor(observation))
                As.append(action) #record action for training

                #pass action to environment, get next observation
                observation, reward, terminated, truncated, _ = environment.step(action)
                observation = tf.expand_dims(tf.convert_to_tensor(observation), 0)
                Rs.append(float(reward)) #record reward for training
                Ss.append(observation) #record observation for training
                
                agent.handleStep(terminated or truncated, Ss, As, Rs, callbacks=[
                    #tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda _, logs: Ls.append(logs.get("loss"))) #for logging loss as a metric (not used atm)
                ])
            #episode finished
            sumRs = sum(Rs)
            metrics[i * N_METRICS + 0] = sumRs
            print(agentType.__name__+": Episode "+str(i)+" Done (r = "+str(sumRs)+", ε = "+str(round(agent.epsilon,2))+")")
            rngSeed+=1
            Ss.clear()
            As.clear()
            Rs.clear()
        #finished training this agent
        #write the model weights to file
        weightsPath = f"checkpoints\\{type(environment).__name__}_{type(agent).__name__}_seed{RNG_SEED}.tf"
        agent.save_weights(weightsPath, overwrite=True)
    except Exception as e:
        #inform parent about the crash and then keep crashing
        anyProcessFailed.value = 1
        raise e

if __name__ == "__main__":
    environment = MazeEnv
    #in order for the child processes to pass them to the agent constructor, all args must be specified, even if they're unused by this agent
    agentConfigs: list[tuple[Agent,tuple]]
    agentConfigs = [
        (PPOAgent, (
            [layers.Conv2D(4,3,(1,1)), layers.MaxPool2D((2,2)), layers.Flatten(), layers.Dense(16), layers.Dense(16), layers.Dense(16), layers.Dense(16), layers.Dense(16), layers.Dense(16)],
            .00001, #learning rate
            .2, #epsilon
            1, #epsilon decay
            .99, #discount rate
            5, # entropyWeight, 
            2, # criticWeight, 
            100, # tMax,
            .2, # interval, 
            0, # replayMemoryCapacity, 
            0, # replayFraction
        )),
        # (AdvantageActorCriticAgent, (
        #     [layers.Conv2D(4,3,(1,1)), layers.Flatten()],
        #     .00015, #learning rate
        #     .2, #epsilon
        #     1, #epsilon decay
        #     .99, #discount rate
        #     5, # entropyWeight, 
        #     2, # criticWeight, 
        #     10, # tMax,
        #     0, # interval, 
        #     0, # replayMemoryCapacity, 
        #     0, # replayFraction
        # )),
        # (ActorCriticAgent, (
        #     [layers.Conv2D(4,2,(1,1)), layers.MaxPool2D((2,2)), layers.Flatten(), layers.Dense(16)],
        #     .0001, #learning rate
        #     .2, #epsilon
        #     1, #epsilon decay
        #     .99, #discount rate
        #     5, # entropyWeight, 
        #     2, # criticWeight, 
        #     0, # tMax, 
        #     0, # interval, 
        #     1000, # replayMemoryCapacity, 
        #     10, # replayFraction
        # )),
        # (DQNAgent, (
        #     [layers.Conv2D(4,3,(1,1)), layers.Flatten()],
        #     .00015, #learning rate
        #     .2, #epsilon
        #     1, #epsilon decay
        #     .99, #discount rate
        #     0, # entropyWeight, 
        #     0, # criticWeight, 
        #     0, # tMax, 
        #     0, # interval, 
        #     1000, # replayMemoryCapacity, 
        #     10, # replayFraction
        # )),
        # (REINFORCEAgent, (
        #     [layers.Conv2D(4,3,(1,1)), layers.Flatten()],
        #     .00015, #learning rate
        #     .2, #epsilon
        #     1, #epsilon decay
        #     .99, #discount rate
        #     5, # entropyWeight, 
        #     0, # criticWeight, 
        #     0, # tMax, 
        #     0, # interval, 
        #     0, # replayMemoryCapacity, 
        #     0, # replayFraction
        # ))
    ]

    metrics = []
    for i in range(len(agentConfigs)):
        metrics.append(mp.Array('d', N_EPISODES * N_METRICS))
    startDatetime = datetime.datetime.now()
    anyProcessFailed = mp.Value("i",0) #for stopping the other processes if one fails
    #Pool doesn't work here, throws due to the shared metrics variables ^
    os.makedirs("checkpoints", exist_ok=True)
    processes: list[mp.Process]
    processes = []
    for i in range(len(agentConfigs)):
        process = mp.Process(target=train, args=[
            agentConfigs[i][0],
            agentConfigs[i][1],
            metrics[i],
            anyProcessFailed
        ])
        processes.append(process)
        process.start()
    for process in processes:
        if anyProcessFailed.value==1: #if any process failed, the whole batch is ruined.
            process.kill() #kill 'em all
        process.join()  
        #finished training all agents on this environment
    if anyProcessFailed.value==1:
        print("Training crashed.")
    else:
        nonSharedMetrics = []
        for i in range(len(agentConfigs)):
            nonSharedMetrics.append([])
            for j in range(N_METRICS):
                nonSharedMetrics[i].append([])
                for k in range(N_EPISODES):
                    nonSharedMetrics[i][j].append(metrics[i][k * N_METRICS + j])
        #finished training all environments
        #write the metrics to file
        metricsDir = os.path.dirname(os.path.abspath(__file__)) + "\\metrics"
        os.makedirs(metricsDir, exist_ok=True)
        metadataAgents = [agentConfig[0].__name__ for agentConfig in agentConfigs]
        metadataEnvironments = [environment.__name__]
        np.savez(
            metricsDir + f"\\metrics_{environment.__name__}_epochs{N_EPISODES}_seed{RNG_SEED}.npz",
            agents=metadataAgents,
            environments=metadataEnvironments,
            nEpisodes = [N_EPISODES],
            data=nonSharedMetrics
        )

        def plot(yss, j, label):
            for i in range(len(agentConfigs)):
                ys = yss[i][j]
                x = range(len(ys))

                #smooth the curve
                smoothedYs = []
                window = []
                windowSize = N_EPISODES/10
                for y in ys:
                    window.append(y)
                    if len(window)>windowSize:
                        window.pop(0)
                    smoothedYs.append(sum(window)/windowSize)
                plt.plot(x,smoothedYs, label=label + "(" + agentConfigs[i][0].__name__ + ")")
                plt.title(environment.__name__)
                plt.legend()

        #plot metrics
        plot(nonSharedMetrics, 0, "reward")

        print("Finished training after", datetime.datetime.now().__sub__(startDatetime))
        plt.show()
        input("press any key to continue") #because pyplot fails to show sometimes