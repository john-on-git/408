import matplotlib.pyplot as plt
import numpy as np


npz = np.load("metrics\metrics_2024.03.13-01.43.08.npz")
environments = npz["environments"]
agents = npz["agents"]
nEpochs = npz["nEpochs"][0]
metrics = npz["data"]

def plot(target, yss, i,m, label):
        for j in range(len(agents)):
            ys = []
            for k in range(nEpochs):
                ys.append(yss[i][j][k][m])
                
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
            target.plot(x,smoothedYs, label=label + "(" + agents[j] + ")")
            #m, c = np.polyfit(x,ys,1)
            #plt.plot(m*x + c) #line of best fit
            target.title.set_text(environments[i])
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