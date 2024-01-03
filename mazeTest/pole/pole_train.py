import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gym
from pole_agents import *

if __name__ == "__main__":
    RNG_SEED_INIT=42
    TARGET_EPOCHS_INIT = 25

    agents = [
        #REINFORCEAgent(learningRate=.01, discountRate=0.8, baseline=0),
        MonteCarloAgent(learningRate=.75, discountRate=.8, replayMemoryCapacity=1000),
        #SARSAAgent(learningRate=0.5, discountRate=.8, replayMemoryCapacity=5000, epsilon=0.8, kernelSeed=RNG_SEED_INIT),
        #  DQNAgent(learningRate=0.00025, discountRate=.8, replayMemoryCapacity=50000, replayFraction=200, epsilon=0.25, kernelSeed=RNG_SEED_INIT),
        RandomAgent()
    ]
    agents[0].load_weights("checkpoints\PoleDQNAgent.tf") #TODO

    trainingRunning = True
    targetEpochs = TARGET_EPOCHS_INIT
    resetEpochs = 0
    yss = []
    for i in range(len(agents)):
        yss.append(list())
    
    random.seed(RNG_SEED_INIT)
    tf.random.set_seed(RNG_SEED_INIT)
    np.random.seed(RNG_SEED_INIT)

    while trainingRunning:
        for i in range(len(agents)):
            env = gym.make('CartPole-v1')
            rngSeed=RNG_SEED_INIT
            env.action_space.seed(rngSeed)
            observation, _ = env.reset(seed=rngSeed)
            observation = tf.expand_dims(tf.convert_to_tensor(observation),0)
            epochs = resetEpochs
            print("Training new ", type(agents[i]).__name__)
            while epochs < targetEpochs: #for each epoch
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
                    nextObservation = tf.expand_dims(nextObservation,0)

                    agents[i].handleStep(terminated or truncated, Ss, As, Rs)
                    
                    if terminated or truncated:
                        nextObservation, _ = env.reset(seed=rngSeed)
                        nextObservation = tf.convert_to_tensor(nextObservation)
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
            path = "checkpoints\\" + "Pole" + type(agent).__name__ + ".tf"
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
            resetEpochs=epochs+1
            targetEpochs+=int(n)
        else:
            trainingRunning = False
            env.close()
            
    
    exit()
   