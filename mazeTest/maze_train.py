import datetime
import random
from maze_env import MazeEnv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from models.REINFORCEModel import REINFORCEModel
from models.PPOModel import PPOModel

if __name__ == "__main__":
    RNG_SEED_INIT=42
    TARGET_EPOCHS_INIT = 1000

    random.seed(RNG_SEED_INIT)
    tf.random.set_seed(RNG_SEED_INIT)
    np.random.seed(RNG_SEED_INIT)

    models = [
        PPOModel(.75, .75, epsilonFraction=10),
        REINFORCEModel(.75)
    ]
    #models[0].load_weights("checkpoints\MazeQLearningModelWithExperienceReplay_20231218_234226.tf") #TODO
    trainingRunning = True
    targetEpochs = TARGET_EPOCHS_INIT
    yss = []
    while trainingRunning:
        for model in models:
            env = MazeEnv(nCoins=10, startPosition="random")
            
            rngSeed = RNG_SEED_INIT
            observation, _ = env.reset(seed=rngSeed)
            observation = tf.expand_dims(tf.convert_to_tensor(observation),0)

            rewardsOverall = []
            epochs = 0
            print("Training new ", type(model).__name__)
            while epochs < targetEpochs: #for each epoch
                Ss = []
                As = []
                Rs = []
                t = 0
                epochRunning = True
                while epochRunning: #for each time step in epoch
                    Ss.append(observation) #record observation for training

                    #prompt agent
                    action = model.act(tf.convert_to_tensor(observation))
                    As.append(action) #record observation for training

                    #pass action to env, get next observation
                    nextObservation, reward, terminated, truncated, _ = env.step(action)
                    Rs.append(float(reward)) #record observation for training
                    
                    nextObservation = tf.convert_to_tensor(nextObservation)
                    nextObservation = tf.expand_dims(nextObservation,0)

                    model.handleStep(terminated or truncated, Ss, As, Rs)
                    
                    if terminated or truncated:
                        nextObservation, _ = env.reset(seed=rngSeed)
                        nextObservation = tf.convert_to_tensor(nextObservation)
                        nextObservation = tf.expand_dims(nextObservation,0)
                        
                        #calc overall reward for graph
                        rewardsOverall.append(sum(Rs))
                        Ss = []
                        As = []
                        Rs = []
                        epochRunning = False
                        print("Epoch ", epochs, " Done (reward ", rewardsOverall[-1], ")", sep="")
                    observation = nextObservation
                    t+=1
                epochs+=1
                rngSeed+=1
            #finished training this model
            yss.append(rewardsOverall)
        
        #finished training all models
            
        #save the models
        for model in models:
            path = "checkpoints\\" + type(model).__name__ + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".tf"
            #path = "checkpoints\\mazeModel.tf"
            model.save_weights(path, overwrite=True)
        
        #plot return over time for each model
        for i in range(len(models)):
            ys = yss[i]
            x = range(len(ys))
            
            #smooth the curve
            smoothedYs = []
            window = []
            windowSize = max(len(ys)/200, 1)
            for y in ys:
                window.append(y)
                if len(window)>windowSize:
                    window.pop()
                smoothedYs.append(sum(window)/windowSize)
            
            plt.plot(x,smoothedYs, label=type(models[i]).__name__)
            #m, c = np.polyfit(x,ys,1)
            #plt.plot(m*x + c) #line of best fit
        plt.legend()
        plt.show()
        
        #prompt to continue training
        
        n = input("enter number to extend training, non-numeric to end\n")
        if(n.isnumeric()):
            targetEpochs+=int(n)
        else:
            trainingRunning = False
            env.close()
            
    
    exit()
   