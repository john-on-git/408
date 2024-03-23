import random
import numpy as np
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
from environments import Environment, MazeEnv, TagEnv, TTTEnv
from agents import *
import datetime
import os

if __name__ == "__main__":
    RNG_SEED_INIT = 0 #fixed RNG for replicability.
    N_EPISODES = 1000 #number of episodes to train for
    GRAPH_WINDOW_SIZE = N_EPISODES/10

    startDatetime = datetime.datetime.now()
    
    environments: list[Environment]
    environments = [
        MazeEnv(startPosition=[(0,0),(5,0),(0,5),(5,5)]),
        TagEnv(seekerSpread=0),
        TTTEnv()
    ]
    mazeAgents = [ #discount rate too low?
        PPOAgent(
            actionSpace=environments[0].actionSpace,
            hiddenLayers=[layers.Flatten(), layers.Dense(32, activation=tf.nn.relu)],
            validActions=environments[0].validActions,
            learningRate=.00001, #.0001, .001 crashed
            discountRate=.99,
            epsilon=.5,
            epsilonDecay=.9999,
            criticWeight=5,
            entropyWeight=1
        ),
        # AdvantageActorCriticAgent(
        #     actionSpace=environments[0].actionSpace,
        #     hiddenLayers=[layers.Flatten(), layers.Dense(328, activation=tf.nn.relu)],
        #     validActions=environments[0].validActions,
        #     learningRate=.0001,
        #     discountRate=.95,
        #     epsilon=.5,
        #     epsilonDecay=.9999,
        #     criticWeight=5,
        #     entropyWeight=1
        # ),
        # ActorCriticAgent(
        #     actionSpace=environments[0].actionSpace,
        #     hiddenLayers=[layers.Flatten(), layers.Dense(32, activation=tf.nn.relu)],
        #     validActions=environments[0].validActions,
        #     learningRate=.0001,
        #     discountRate=.95,
        #     epsilon=.5,
        #     epsilonDecay=.995,
        #     replayMemoryCapacity=10000,
        #     replayFraction=100
        # ),
        # DQNAgent(
        #     actionSpace=environments[0].actionSpace,
        #     hiddenLayers=[layers.Flatten(), layers.Dense(32, activation=tf.nn.relu)],
        #     validActions=environments[0].validActions,
        #     learningRate=.0001,
        #     discountRate=.95,
        #     epsilon=.5,
        #     epsilonDecay=.9999,
        #     replayMemoryCapacity=10000,
        #     replayFraction=100
        # ),
        # REINFORCEAgent(
        #     actionSpace=environments[0].actionSpace,
        #     hiddenLayers=[layers.Flatten(), layers.Dense(8, activation=tf.nn.relu)],
        #     validActions=environments[0].validActions,
        #     learningRate=.0001,
        #     discountRate=.95,
        #     epsilon=.5,
        #     epsilonDecay=.9999,
        # ),
    ]
    tagAgents = [
        PPOAgent(
            actionSpace=environments[1].actionSpace,
            hiddenLayers=[layers.Flatten(), layers.Dense(16, activation=tf.nn.sigmoid)],
            validActions=environments[1].validActions,
            learningRate=.001, #best so far
            discountRate=.95,
            epsilon=0,
            criticWeight=2,
            interval=0.3
        ),
        DQNAgent(
            actionSpace=environments[1].actionSpace,
            hiddenLayers=[layers.Flatten(), layers.Dense(16, activation=tf.nn.sigmoid)],
            validActions=environments[1].validActions,
            learningRate=.001,
            discountRate=.95,
            epsilon=0,
            replayMemoryCapacity=1000,
            replayFraction=10
        ),
        AdvantageActorCriticAgent(
            actionSpace=environments[1].actionSpace,
            hiddenLayers=[layers.Flatten(), layers.Dense(16, activation=tf.nn.sigmoid)],
            validActions=environments[1].validActions,
            learningRate=.001,
            discountRate=.95,
            epsilon=0,
            criticWeight=2
        ),
        ActorCriticAgent(
            actionSpace=environments[1].actionSpace,
            hiddenLayers=[layers.Flatten(), layers.Dense(16, activation=tf.nn.sigmoid)],
            validActions=environments[1].validActions,
            learningRate=.001,
            discountRate=.95,
            epsilon=0,
            replayFraction=10
        ),
        REINFORCEAgent(
            actionSpace=environments[1].actionSpace,
            hiddenLayers=[layers.Flatten(), layers.Dense(16, activation=tf.nn.sigmoid)],
            validActions=environments[1].validActions,
            learningRate=.001,
            discountRate=.95,
            epsilon=0,
            epsilonDecay=.99
        ),
    ]
    TTTAgents = [
        PPOAgent(
            actionSpace=environments[2].actionSpace,
            hiddenLayers=[layers.Flatten(), layers.Dense(16, activation=tf.nn.sigmoid)],
            validActions=environments[2].validActions,
            learningRate=.001,
            discountRate=.9,
            epsilon=.25,
            epsilonDecay=.99,
            criticWeight=5,
            entropyWeight=1
        ),
        AdvantageActorCriticAgent(
            actionSpace=environments[2].actionSpace,
            hiddenLayers=[layers.Flatten(), layers.Dense(16, activation=tf.nn.sigmoid)],
            validActions=environments[2].validActions,
            learningRate=.001,
            discountRate=.9,
            epsilon=.25,
            epsilonDecay=.99,
            criticWeight=10
        ),
        ActorCriticAgent(
            actionSpace=environments[2].actionSpace,
            hiddenLayers=[layers.Flatten(), layers.Dense(16, activation=tf.nn.sigmoid)],
            validActions=environments[2].validActions,
            learningRate=.001,
            discountRate=.9,
            epsilon=.25,
            epsilonDecay=.99,
            replayFraction=10
        ), 
        DQNAgent(
            actionSpace=environments[2].actionSpace,
            hiddenLayers=[layers.Flatten(), layers.Dense(16, activation=tf.nn.sigmoid)],
            validActions=environments[2].validActions,
            learningRate=.001,
            discountRate=.9,
            epsilon=.5,
            epsilonDecay=.99,
            replayMemoryCapacity=1000,
            replayFraction=10
        ),
        REINFORCEAgent(
            actionSpace=environments[2].actionSpace,
            hiddenLayers=[layers.Flatten(), layers.Dense(16, activation=tf.nn.sigmoid)],
            validActions=environments[2].validActions,
            learningRate=.001,
            discountRate=.9,
            epsilon=.5,
            epsilonDecay=.99
        ),
    ]

    
    environments = [
        environments[0],
        #environments[1],
        #environments[2]
    ]
    agents: list[list[Agent]]
    agents = [
        mazeAgents,
        #tagAgents,
        #TTTAgents
    ]
    #assert that there are an equal number of agents for all enviroments
    assert len(environments) == len(agents)
    for xs in agents:
        assert len(xs) == len(agents[0])
    
    metrics = np.ndarray(shape=(len(environments), len(agents[0]), N_EPISODES, 2))
    for i in range(len(environments)):
        agents: list[list[Agent]]
        for j in range(len(agents[i])):
            print("Training new", type(environments[i]).__name__,"/",type(agents[i][j]).__name__)

            rngSeed=RNG_SEED_INIT
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
                    action = agents[i][j].act(tf.convert_to_tensor(observation))
                    As.append(action) #record action for training

                    #pass action to environment, get next observation
                    observation, reward, terminated, truncated, _ = environments[i].step(action)
                    observation = tf.expand_dims(tf.convert_to_tensor(observation), 0)
                    Rs.append(float(reward)) #record reward for training
                    Ss.append(observation) #record observation for training
                    
                    agents[i][j].handleStep(terminated or truncated, Ss, As, Rs, callbacks=[
                        #tf.keras.callbacks.LambdaCallback(on_episode_end=lambda _, logs: Losses.append(logs["loss"])) #for logging loss as a metric (not used atm)
                    ])
                #episode finished
                metrics[i][j][k][0] = sum(Rs)
                metrics[i][j][k][1] = datetime.datetime.now().timestamp()
                print("Episode ", k, " Done (r = ", metrics[i][j][k][0],", ε = ", round(agents[i][j].epsilon,2), ")", sep="")
                rngSeed+=1
            #finished training this agent
            #write the model weights to file
            weightsPath = "checkpoints\\" + type(environments[i]).__name__ + "_" + type(agents[i][j]).__name__ + ".tf"
            agents[i][j].save_weights(weightsPath, overwrite=True)
        #finished training all agents on this environment
    #finished training all environments
    #write the metrics to file
    metricsDir = os.path.dirname(os.path.abspath(__file__)) + "\\metrics"
    os.makedirs(metricsDir, exist_ok=True)
    metadataAgents = [[type(x).__name__ for x in xs] for xs in agents]
    metadataEnvironments = [type(x).__name__ for x in environments]
    np.savez(
        metricsDir + "\\metrics_" + datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S"),
        agents=metadataAgents,
        environments=metadataEnvironments,
        nEpisodes = [N_EPISODES],
        data=metrics
    )

    def plot(target, yss, i,m, label):
        match type(environments[i]).__name__:
            case "MazeEnv":
                target.axhline(y=4767/2, color="grey")
            case "TagEnv":
                target.axhline(y=565/2, color="grey")
            case "TTTEnv":
                target.axhline(y=1000/2, color="lightgrey")
        for j in range(len(agents[i])):
            ys = []
            for k in range(N_EPISODES):
                ys.append(yss[i][j][k][m])

            x = range(len(ys))

            #smooth the curve
            smoothedYs = []
            window = []
            for y in ys:
                window.append(y)
                if len(window)>GRAPH_WINDOW_SIZE:
                    window.pop(0)
                smoothedYs.append(sum(window)/GRAPH_WINDOW_SIZE)
            target.plot(x,smoothedYs, label=label + "(" + type(agents[i][j]).__name__ + ")")
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

    print("Finished training after", datetime.datetime.now().__sub__(startDatetime))
    plt.show()

    input("press any key to continue") #because pyplot fails to show sometimes
    #prompt to continue training
    
    #n = input("enter number to extend training, non-numeric to end\n")
    #if(n.isnumeric()):
    #    resetEpisodes=episodes
    #    targetEpisodes+=int(n)
    #else:
    #    trainingRunning = False
    #    environments[i].close()