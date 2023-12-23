import datetime
import random
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PoleREINFORCEmodel import REINFORCEModel, REINFORCEModel2
from PoleDQNModel import PoleDQNModel
if __name__ == "__main__":
    RNG_SEED_INIT=42
    TARGET_EPOCHS_INIT = 1000

    random.seed(RNG_SEED_INIT)
    tf.random.set_seed(RNG_SEED_INIT)
    np.random.seed(RNG_SEED_INIT)

    models = [
        PoleDQNModel(
            learningRate=.5,
            discountRate=.99,
            replayMemoryCapacity=10000,
        )
    ]
    for model in models:
        rngSeed=RNG_SEED_INIT
        targetEpochs = TARGET_EPOCHS_INIT
        env = gym.make('CartPole-v1')
        env.action_space.seed(rngSeed)
        
        
        observation, _ = env.reset(seed=rngSeed)
        observation = tf.expand_dims(tf.convert_to_tensor(observation),0)

        rewardsOverall = []
        epochs = 0
        print("Training new ", type(model).__name__)
        while True:
            while epochs < targetEpochs:
                Ss = []
                As = []
                Rs = []
                t = 0
                stop = False
                while not stop:
                    Ss.append(observation) #record observation for training

                    #prompt agent
                    action = model.act(tf.convert_to_tensor(observation))
                    As.append(action) #record observation for training

                    #pass action to env, get next observation
                    nextObservation, reward, terminated, truncated, _ = env.step(action)
                    Rs.append(reward) #record observation for training
                    
                    nextObservation = tf.convert_to_tensor(nextObservation)
                    nextObservation = tf.expand_dims(nextObservation,0)

                    model.handleStep(terminated or truncated, Ss, As, Rs,) #template method pattern 🤓
                    
                    if terminated or truncated:
                        nextObservation, _ = env.reset(seed=rngSeed)
                        nextObservation = tf.convert_to_tensor(nextObservation)
                        nextObservation = tf.expand_dims(nextObservation,0)
                        
                        #calc overall reward for graph
                        rewardsOverall.append(sum(Rs))
                        Ss = []
                        As = []
                        Rs = []
                        stop = True
                    observation = nextObservation
                    t+=1
                epochs+=1
                rngSeed+=1
                print("Epoch ", epochs, " Done (duration ", t, ")", sep="")

            

            n = input("enter number to extend training, non-numeric to end\n")
            if(n.isnumeric()):
                targetEpochs+=int(n)
            else:
                path = "checkpoints\\" + type(model).__name__ + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".tf"
                #path = "checkpoints\\" + type(model).__name__ + "_" + str(TARGET_EPOCHS_INIT) + ".tf"
                model.save_weights(path, overwrite=True)
                #plot return over time
                x = range(len(rewardsOverall))
                y = rewardsOverall
                plt.scatter(x,y, label=type(model).__name__)
                m, c = np.polyfit(x,y,1)
                plt.plot(m*x + c) #line of best fit
                env.close()
                break
    
    plt.legend()
    plt.show()        
    exit()
   