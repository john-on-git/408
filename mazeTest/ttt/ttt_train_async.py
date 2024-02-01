import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ttt_env import TTTEnv
from ttt_agents import *
from multiprocessing import Process, Manager
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

def train(rngSeed, agentType, weightCollector, yss, printLock, targetEpochs):
    agent = REINFORCE_MENTAgent(learningRate=0.01, discountRate=.9)
    #mutex lock?
    agent.load_weights("checkpoints\\TTT" + agentType.__name__ + ".tf")
    
    random.seed(rngSeed)
    tf.random.set_seed(rngSeed)
    np.random.seed(rngSeed)
    env = TTTEnv(opponent=SearchAgent(epsilon=.25))
    observation, _ = env.reset(seed=rngSeed)
    observation = tf.expand_dims(tf.convert_to_tensor(observation),0)
    epochs = 0

    while epochs < targetEpochs: #for each epoch
        Ss = []
        As = []
        Rs = []
        epochRunning = True
        while epochRunning: #for each time step in epoch
            Ss.append(observation) #record observation for training

            #prompt agent
            action = agent.act(tf.convert_to_tensor(observation))
            As.append(action) #record observation for training

            #pass action to env, get next observation
            nextObservation, reward, terminated, truncated, _ = env.step(action)

            Rs.append(float(reward)) #record observation for training
            
            nextObservation = tf.convert_to_tensor(nextObservation)
            nextObservation = tf.expand_dims(nextObservation,0)

            agent.handleStep(terminated or truncated, Ss, As, Rs)
            
            if terminated or truncated:
                nextObservation, _ = env.reset(seed=rngSeed)
                nextObservation = tf.convert_to_tensor(nextObservation)
                nextObservation = tf.expand_dims(nextObservation,0)
                
                yss.append(sum(Rs)) #calc overall reward for graph
                Ss = []
                As = []
                Rs = []
                epochRunning = False
                printLock.acquire()
                print("Epoch ", epochs, " Done (r = ", yss[-1],", εA ≈ ", round(agent.epsilon, 2),", εO ≈ ",round(env.opponent.epsilon, 2),")", sep="")
                printLock.release()
            observation = nextObservation
        epochs+=1
        rngSeed+=1

if __name__ == "__main__":
    N_CHILD_PROCESSES = 12
    N_EPOCHS = 500
    N_STEPS_BETWEEN_TARGET_UPDATES = 50
    RNG_SEED_INIT=42
    RNG_STEP = 100000
    with Manager() as manager:
        manager.register("list", list)
        collector = manager.list()
        printLock = manager.Lock()
        agents = [
            #DQNAgent(learningRate=.001, discountRate=.95, replayMemoryCapacity=1000, epsilon=0.25, epsilonDecay=1)
            #RandomAgent()
            REINFORCE_MENTAgent(learningRate=0.01, discountRate=.9)
        ]
        yss = [] #list of rewards each epoch, for each agent
        for agent in agents:
            yss.append(manager.list())
        rngSeed = RNG_SEED_INIT
        
        for i in range(len(agents)):
            print("Training new ", type(agents[i]).__name__)
            for j in range(int(N_EPOCHS/N_STEPS_BETWEEN_TARGET_UPDATES)): #train for a bit, then merge into target network
                processes = []
                for _ in range(N_CHILD_PROCESSES): #create child processes
                    process = Process(target=train, args=[rngSeed, type(agents[i]), collector, yss[i], printLock, N_STEPS_BETWEEN_TARGET_UPDATES])
                    processes.append(process)
                    process.start()
                    rngSeed+=RNG_STEP

                for process in processes: #wait for children to finish
                    process.join()
                
                #take average of delta-weights
                for i in range(len(collector)):
                    pass #collector[i]/=N_CHILD_PROCESSES
                #apply weights
                agents[i].optimizer.apply_gradients(zip(
                    collector,
                    agents[i].trainable_weights
                ))
                for i in range(len(collector)): #reset
                    collector[i] = None
            #write checkpoint
            agents[i].save_weights("checkpoints\\TTT" + type(agents[i]).__name__ + ".tf", overwrite=True) #save the agent
        
            #plot return over time for each agent
            ys = yss[i]
            x = range(len(ys))
            
            #smooth the curve
            smoothedYs = []
            window = []
            windowSize = max(int(len(ys)/20), 1)
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
        input("press any key to continue")