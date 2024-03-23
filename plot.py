import matplotlib.pyplot as plt
import numpy as np

npz = np.load("metrics/metrics_2024.03.23-05.17.36.npz")
environments = npz["environments"]
agents = npz["agents"]
nEpisodes = npz["nEpisodes"][0]
metrics = npz["data"]

def plot(target, i,m, label):
    #baseline/targets
    match environments[i]:
        case "MazeEnv":
            target.axhline(y=4767/2, color="lightgrey")
        case "TagEnv":
            target.axhline(y=565/2, color="lightgrey")
        case "TTTEnv":
            target.axhline(y=1000/2, color="lightgrey")
    for j in range(len(agents[i])):
        ys = []
        for k in range(nEpisodes):
            ys.append(metrics[i][j][k][m])
            
        x = range(len(ys))
        
        #smooth the curve
        smoothedYs = []
        window = [] 
        windowSize = 500
        for y in ys:
            window.append(y)
            if len(window)>windowSize:
                window.pop(0)
            smoothedYs.append(sum(window)/windowSize)
        target.plot(x,smoothedYs, label=agents[i][j])
        target.title.set_text(environments[i])
        target.legend()
        plt.xlabel("Number of Episodes")
        plt.ylabel(label + " (smoothed)")

#plot return over time for envs/agents
iter = [0]
fig, axs = plt.subplots(nrows=len(iter))
#print
if len(iter) == 1: #axs is a list, unless the first dimension would be length 1, then it isn't. account for that.
    plot(axs, iter[0],0, "reward")
else:
    i = 0
    for j in iter:
        plot(axs[i], j,0, "reward")
        i+=1
#plt.savefig('img/CHANGEME.png')
plt.show()