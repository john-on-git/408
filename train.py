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
    RNG_SEED_INIT=42
    N_TRAINING_EPOCHS = 250
    N_AGENTS = 1

    environments: list[Environment]
    environments = [
        #MazeEnv(nCoins=10),
        TagEnv(),
        #TTTEnv()
    ]
    metrics = np.ndarray(shape=(len(environments), N_AGENTS, N_TRAINING_EPOCHS, 2))
    for i in range(len(environments)):
        random.seed(RNG_SEED_INIT)
        tf.random.set_seed(RNG_SEED_INIT)
        np.random.seed(RNG_SEED_INIT)

        agents: list[Agent]
        agents = [
            # REINFORCEAgent(
            #     actionSpace=environments[i].ACTION_SPACE,
            #     hiddenLayers=[layers.Flatten(), layers.Dense(16, activation=tf.nn.sigmoid),layers.Dense(32, activation=tf.nn.sigmoid)],
            #     validActions=environments[i].validActions,
            #     learningRate=.001,
            #     discountRate=.95,
            #     epsilon=0.25,
            #     epsilonDecay=.9
            # ),
            # REINFORCE_MENTAgent(
            #     actionSpace=environments[i].ACTION_SPACE,
            #     hiddenLayers=[layers.Flatten(), layers.Dense(16, activation=tf.nn.sigmoid),layers.Dense(32, activation=tf.nn.sigmoid)],
            #     validActions=environments[i].validActions,
            #     learningRate=.001,
            #     discountRate=.95,
            #     epsilon=0.25,
            #     epsilonDecay=.9
            # ),
            # PPOAgent(
            #     actionSpace=environments[i].ACTION_SPACE,
            #     hiddenLayers=[layers.Flatten(), layers.Dense(16, activation=tf.nn.sigmoid),layers.Dense(32, activation=tf.nn.sigmoid)],
            #     validActions=environments[i].validActions,
            #     learningRate=.001,
            #     discountRate=.95,
            #     epsilon=0.25,
            #     epsilonDecay=.9
            # ),
            # DQNAgent(
            #    actionSpace=environments[i].ACTION_SPACE,
            #    hiddenLayers=[layers.Flatten(), layers.Dense(16, activation=tf.nn.sigmoid),layers.Dense(32, activation=tf.nn.sigmoid)],
            #    validActions=environments[i].validActions,
            #    learningRate=.001,
            #    discountRate=.95,
            #    epsilon=0.25,
            #    epsilonDecay=.9
            # ),
            # SARSAAgent(
            #     actionSpace=environments[i].ACTION_SPACE,
            #     hiddenLayers=[layers.Flatten(), layers.Dense(16, activation=tf.nn.sigmoid),layers.Dense(32, activation=tf.nn.sigmoid)],
            #     validActions=environments[i].validActions,
            #     learningRate=.001,
            #     discountRate=.95,
            #     epsilon=0.25,
            #     epsilonDecay=.9
            # ),
            # ActorCriticAgent(
            #     actionSpace=environments[i].ACTION_SPACE,
            #     hiddenLayers=[layers.Flatten(), layers.Dense(16, activation=tf.nn.sigmoid),layers.Dense(32, activation=tf.nn.sigmoid)],
            #     validActions=environments[i].validActions,
            #     learningRate=.001,
            #     discountRate=.95,
            #     epsilon=0.25,
            #     epsilonDecay=.9
            # ),
            PPOAgent(
                actionSpace=environments[i].ACTION_SPACE,
                hiddenLayers=[layers.Flatten(), layers.Dense(16, activation=tf.nn.sigmoid),layers.Dense(32, activation=tf.nn.sigmoid)],
                validActions=environments[i].validActions,
                learningRate=.001,
                discountRate=.95,
                epsilon=0.5,
                epsilonDecay=.9
            ),
            # AdvantageActorCriticAgent(
            #     actionSpace=environments[i].ACTION_SPACE,
            #     hiddenLayers=[layers.Flatten(), layers.Dense(16, activation=tf.nn.sigmoid),layers.Dense(32, activation=tf.nn.sigmoid)],
            #     validActions=environments[i].validActions,
            #     learningRate=.001,
            #     discountRate=.95,
            #     epsilon=0.5,
            #     epsilonDecay=.9,
            #     tMax=1000
            # ),
        ]
        assert len(agents) == N_AGENTS
        for j in range(len(agents)):
            print("Training new", type(environments[i]).__name__,"/",type(agents[j]).__name__)

            rngSeed=RNG_SEED_INIT
            observation, _ = environments[i].reset(rngSeed)
            observation = tf.expand_dims(tf.convert_to_tensor(observation),0)
            start = time.time()
            epochs = 0
            for k in range(N_TRAINING_EPOCHS):
                Ss = []
                As = []
                Rs = []
                Losses = []
                epochRunning = True
                while epochRunning: #for each time step in epoch
                    Ss.append(observation) #record observation for training

                    #prompt agent
                    action = agents[j].act(tf.convert_to_tensor(observation))
                    As.append(action) #record observation for training

                    #pass action to environment, get next observation
                    nextObservation, reward, terminated, truncated, _ = environments[i].step(action)
                    Rs.append(float(reward)) #record observation for training
                    
                    nextObservation = tf.convert_to_tensor(nextObservation)
                    nextObservation = tf.expand_dims(nextObservation,0)

                    agents[j].handleStep(terminated or truncated, Ss, As, Rs, callbacks=[
                        #tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda _, logs: Losses.append(logs["loss"]))
                    ])
                    
                    if terminated or truncated:
                        nextObservation, _ = environments[i].reset(rngSeed)
                        nextObservation = tf.convert_to_tensor(nextObservation)
                        nextObservation = tf.expand_dims(nextObservation,0)

                        metrics[i][j][k][0] = sum(Rs)
                        metrics[i][j][k][1] = time.time()
                        Ss = []
                        As = []
                        Rs = []
                        epochRunning = False
                        print("Epoch ", epochs, " Done (r = ", metrics[i][j][k][0],", ε = ", round(agents[j].epsilon,2), ")", sep="")
                    observation = nextObservation
                epochs+=1
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
        nEpochs = [N_TRAINING_EPOCHS],
        data=metrics
    )

    def plot(target, yss, i,m, label):
        for j in range(len(agents)):
            ys = []
            for k in range(N_TRAINING_EPOCHS):
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
        
    #prompt to continue training
    
    #n = input("enter number to extend training, non-numeric to end\n")
    #if(n.isnumeric()):
    #    resetEpochs=epochs
    #    targetEpochs+=int(n)
    #else:
    #    trainingRunning = False
    #    environments[i].close()