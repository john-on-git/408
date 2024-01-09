import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from maze_env import MazeEnv
from maze_agents import *

if __name__ == "__main__":
    RNG_SEED_INIT=42
    TARGET_EPOCHS_INIT = 43

    agents = [
        #REINFORCEAgent(learningRate=.01, discountRate=.75, baseline=0),
        #MonteCarloAgent(learningRate=.001, discountRate=.75, replayMemoryCapacity=5000, replayFraction=25, epsilon=.99, epsilonDecay=.9999),
        #SARSAAgent(learningRate=.001, discountRate=.95, replayMemoryCapacity=1000, epsilon=.75, epsilonDecay=.999),
        #DQNAgent(learningRate=.001, discountRate=.75, replayMemoryCapacity=5000, replayFraction=25, epsilon=.99, epsilonDecay=.999),
        ActorCriticAgent(learningRate=.001, discountRate=.75, replayMemoryCapacity=50000, replayFraction=500, epsilon=.99, epsilonDecay=.9999),
        RandomAgent()
    ]

    trainingRunning = True
    targetEpochs = TARGET_EPOCHS_INIT #epochs to train for
    resetEpochs = 0 #epoch count to continue from, when training is extended
    yss = [] #list of rewards each epoch, for each agent
    for i in range(len(agents)):
        yss.append(list())
    
    random.seed(RNG_SEED_INIT)
    tf.random.set_seed(RNG_SEED_INIT)
    np.random.seed(RNG_SEED_INIT)

    while trainingRunning:
        for i in range(len(agents)):
            env = MazeEnv(nCoins=10, startPosition="random")
            rngSeed=RNG_SEED_INIT
            observation, _ = env.reset(seed=rngSeed)
            observation = tf.expand_dims(tf.convert_to_tensor(observation),2)
            observation = tf.expand_dims(tf.convert_to_tensor(observation),0)
            epochs = resetEpochs

            print("Training new ", type(agents[i]).__name__)
            while epochs < targetEpochs: #for each epoch
                agents[i].epsilon*=.99
                Ss = []
                As = []
                Rs = []
                epochRunning = True
                while epochRunning: #for each time step in epoch
                    Ss.append(observation) #record observation for training

                    #prompt agent
                    action = agents[i].act(tf.convert_to_tensor(observation))
                    As.append(action) #record observation for training

                    #pass action to env, get next observation
                    nextObservation, reward, terminated, truncated, _ = env.step(action)

                    Rs.append(float(reward)) #record observation for training
                    
                    nextObservation = tf.convert_to_tensor(nextObservation)
                    nextObservation = tf.expand_dims(nextObservation,2)
                    nextObservation = tf.expand_dims(nextObservation,0)

                    agents[i].handleStep(terminated or truncated, Ss, As, Rs)
                    
                    if terminated or truncated:
                        nextObservation, _ = env.reset(seed=rngSeed)
                        nextObservation = tf.convert_to_tensor(nextObservation)
                        nextObservation = tf.expand_dims(nextObservation,2)
                        nextObservation = tf.expand_dims(nextObservation,0)
                        
                        yss[i].append(sum(Rs)) #calc overall reward for graph
                        Ss = []
                        As = []
                        Rs = []
                        epochRunning = False
                        print("Epoch ", epochs, " Done (r = ", yss[i][-1],", ε ≈ ", round(agents[i].epsilon, 2),")", sep="")
                    observation = nextObservation
                epochs+=1
                rngSeed+=1
            #finished training this agent
        
        #finished training all agents
            
        #save the agents
        for agent in agents:
            #path = "checkpoints\\" + type(agent).__name__ + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".tf"
            path = "checkpoints\\" + "Maze" + type(agent).__name__ + ".tf"
            agent.save_weights(path, overwrite=True)
        
        #plot return over time for each agent
        plt.clf() #clear previous graph
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
            
            plt.plot(x,smoothedYs, label=type(agents[i]).__name__)
            #m, c = np.polyfit(x,ys,1)
            #plt.plot(m*x + c) #line of best fit
        plt.legend()
        plt.show()
        
        #prompt to continue training
        
        n = input("enter number to extend training, non-numeric to end\n")
        if(n.isnumeric()):
            resetEpochs=epochs
            targetEpochs+=int(n)
        else:
            trainingRunning = False
            env.close()
            
    
    exit()
   