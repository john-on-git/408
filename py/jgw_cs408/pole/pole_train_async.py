import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gym
from pole_agents import *
from multiprocessing import Process, Manager, Queue
import os
import time

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

#executed by child processes
def train(rngSeed, agentType, receivedWeights, sentGradients:Queue, lock, yss, running):
    print("eager execution:", tf.executing_eagerly())
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    gradsCollector = []
    agent = agentType(learningRate=0.001, discountRate=.95, epsilon=.75, epsilonDecay=.99, gradsCollector=gradsCollector)
    
    random.seed(rngSeed)
    tf.random.set_seed(rngSeed)
    np.random.seed(rngSeed)
    env = gym.make('CartPole-v1')
    observation, _ = env.reset(seed=rngSeed)
    observation = tf.expand_dims(tf.convert_to_tensor(observation),0)
    epochs = 0

    while True: #for each epoch
        Ss = []
        As = []
        Rs = []
        epochRunning = True
        while epochRunning: #for each time step in epoch
            if running.get()==0:
                exit()
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

                lock.acquire()
                yss.append(sum(Rs)) #calc overall reward for graph
                print(os.getpid(), ": epoch ", epochs, " done (r = ", yss[-1], ")", sep="")
                lock.release()
                
                Ss = []
                As = []
                Rs = []
                epochRunning = False
            observation = nextObservation
        epochs+=1
        rngSeed+=1

        #receive new weights from parent process
        weights = receivedWeights.get()
        for i in range(len(weights)):
            agent.layers[i].set_weights(weights[i])

        #send weights to parent process
        for grads in gradsCollector:
            pickleableGrads = []
            for x in grads:
                pickleableGrads.append(tf.get_static_value(x,partial=False))
            sentGradients.put(pickleableGrads)
        gradsCollector = []
if __name__ == "__main__":
    N_CHILD_PROCESSES = 12
    TRAINING_TIME_SECONDS = 3000
    RNG_SEED_INIT = 42
    RNG_STEP = 100000
    with Manager() as manager:
        manager.register("list", list)
        lock = manager.Lock()
        running = manager.Value("b", 1)
        agents = [
            ParallelDQNAgent(learningRate=0.001, discountRate=.9)
            #RandomAgent()
            #ParallelREINFORCE_MENTAgent(learningRate=0.01, discountRate=.9)
        ]
        yss = [] #list of rewards each epoch, for each agent
        for agent in agents:
            yss.append(manager.list())
        rngSeed = RNG_SEED_INIT
        receivedGradients = Queue()
        sentWeights = manager.Value(list, [])
        kept, discarded = (0,0)
        for i in range(len(agents)):
            #Pass some arbitrary input to the main model. The output is ignored, but TF requires this step to initialize the model correctly.
            obs, _ = gym.make('CartPole-v1').reset()
            _ = agents[i](tf.expand_dims(tf.convert_to_tensor(obs),0))

            print("Training new ", type(agents[i]).__name__)
            processes = []
            for j in range(N_CHILD_PROCESSES): #create child processes#
                process = Process(
                    group=None,
                    target=train,
                    args=[
                        rngSeed+(RNG_STEP*j),
                        type(agents[i]),
                        sentWeights,
                        receivedGradients,
                        lock,
                        yss[i],
                        running
                    ],
                    daemon=True
                )
                processes.append(process)
                process.start()
                rngSeed+=RNG_STEP

            start = time.time()
            while time.time()-start<=TRAINING_TIME_SECONDS:
                #receive and apply all gradients
                updatedWeights = False
                while not receivedGradients.empty():
                    pickleableGrads = receivedGradients.get()
                    if pickleableGrads[0] is not None: #some grads aren't being evaluated? "not in" doesn't work for some reason
                        grads = []
                        for pickleableGrad in pickleableGrads:
                            grads.append(tf.convert_to_tensor(pickleableGrad))
                        agents[i].optimizer.apply_gradients(zip(
                            grads,
                            agents[i].trainable_weights
                        ))
                        updatedWeights = True
                #send out the updated weights
                if updatedWeights:
                    newWeights = []
                    for layer in agents[i].layers:
                        newWeights.append(layer.get_weights())
                    sentWeights.set(newWeights)
            running.set(0)
            while not receivedGradients.empty():
                grads = receivedGradients.get() 
                agents[i].optimizer.apply_gradients(zip(
                    grads,
                    agents[i].trainable_weights
                ))
                del grads[:]
            for process in processes:
                process.join(1)
            #write checkpoint
            agents[i].save_weights("checkpoints\\Pole" + type(agents[i]).__name__ + ".tf", overwrite=True) #save the agent
        
            #plot return over time for each agent
            ys = yss[i]
            
            #smooth the curve
            smoothedYs = []
            window = []
            windowSize = max(int(len(ys)/20), 1)
            for y in ys:
                window.append(y)
                if len(window)>windowSize:
                    window.pop(0)
                smoothedYs.append(sum(window)/windowSize)
            x = range(len(smoothedYs))
            
            plt.plot(x,smoothedYs, label=type(agents[i]).__name__)
            #m, c = np.polyfit(x,ys,1)
            #plt.plot(m*x + c) #line of best fit
        plt.legend()
        plt.show()
        input("press any key to continue")