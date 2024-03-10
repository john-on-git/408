import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from environments import Environment, MazeEnv, TagEnv, TTTEnv
from agents import *
import time

if __name__ == "__main__":
    RNG_SEED_INIT=42
    TRAINING_TIME_SECONDS = 30

    environments: list[Environment]
    environments = [
        #MazeEnv(nCoins=10)
        TagEnv(),
        #TTTEnv()
    ]
    metrics = [] #list of metrics each epoch, for each agent, for each environments[i]

    for i in range(len(environments)):
        random.seed(RNG_SEED_INIT)
        tf.random.set_seed(RNG_SEED_INIT)
        np.random.seed(RNG_SEED_INIT)

        agents: list[Agent]
        agents = [
            ActorCriticAgent(
                actionSpace=environments[i].ACTION_SPACE,
                hiddenLayers=[layers.Flatten(), layers.Dense(16, activation=tf.nn.sigmoid),layers.Dense(32, activation=tf.nn.sigmoid)],
                validActions=environments[i].validActions,
                learningRate=.001,
                discountRate=.95,
                epsilon=0.25,
                epsilonDecay=.9
            ),
            AdvantageActorCriticAgent(
                actionSpace=environments[i].ACTION_SPACE,
                hiddenLayers=[layers.Flatten(), layers.Dense(16, activation=tf.nn.sigmoid),layers.Dense(32, activation=tf.nn.sigmoid)],
                validActions=environments[i].validActions,
                learningRate=.001,
                discountRate=.95,
                epsilon=0.25,
                epsilonDecay=.9
            )
        ]
        metrics.append({"reward":[], "loss":[]})
        for j in range(len(agents)):
            print("Training new", type(environments[i]).__name__,"/",type(agents[j]).__name__)
            metrics[i]["reward"].append(list())
            metrics[i]["loss"].append(list())

            rngSeed=RNG_SEED_INIT
            observation, _ = environments[i].reset(rngSeed)
            observation = tf.expand_dims(tf.convert_to_tensor(observation),0)
            start = time.time()
            epochs = 0
            while time.time()-start<=TRAINING_TIME_SECONDS:
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

                    #pass action to environments[i], get next observation
                    nextObservation, reward, terminated, truncated, _ = environments[i].step(action)
                    Rs.append(float(reward)) #record observation for training
                    
                    nextObservation = tf.convert_to_tensor(nextObservation)
                    nextObservation = tf.expand_dims(nextObservation,0)

                    agents[j].handleStep(terminated or truncated, Ss, As, Rs, callbacks=[
                        tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda _, logs: Losses.append(logs["loss"]))
                    ])
                    
                    if terminated or truncated:
                        nextObservation, _ = environments[i].reset(rngSeed)
                        nextObservation = tf.convert_to_tensor(nextObservation)
                        nextObservation = tf.expand_dims(nextObservation,0)
                        
                        metrics[i]["reward"][j].append(sum(Rs)) #calc overall reward for graph
                        metrics[i]["loss"][j].extend(Losses) #calc overall reward for graph
                        Ss = []
                        As = []
                        Rs = []
                        epochRunning = False
                        print("Epoch ", epochs, " Done (r = ", metrics[i]["reward"][j][-1],")", sep="")
                    observation = nextObservation
                epochs+=1
                rngSeed+=1
            #finished training this agent
        #finished training all agents on this environments[i]ironment
                
        #save the agents
        for agent in agents:
            path = "checkpoints\\tag" + type(agent).__name__ + ".tf"
            agent.save_weights(path, overwrite=True)
        
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

    #plot return over time for envs/agents
    fig, axs = plt.subplots(len(environments))
    if len(environments)>1:
        for i in range(len(environments)): #for each env graph
            plot(axs[i], agents, metrics[i]["reward"], "reward")
    else:
        plot(axs, agents, metrics[0]["reward"], "reward")
    plt.show()
        
    #prompt to continue training
    
    #n = input("enter number to extend training, non-numeric to end\n")
    #if(n.isnumeric()):
    #    resetEpochs=epochs
    #    targetEpochs+=int(n)
    #else:
    #    trainingRunning = False
    #    environments[i]s[i].close()