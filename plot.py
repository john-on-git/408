import matplotlib.pyplot as plt
import numpy as np
import datetime

npz = np.load("metrics/final/metrics_maze_1k_seed0.npz")
environment = npz["environments"][0]
agents = npz["agents"]
nEpisodes = npz["nEpisodes"][0]
metrics = npz["data"]

windowSize = nEpisodes/10

def plot(yss, j, label):
    for i in range(len(agents)):
        ys = yss[i][j]
        x = range(len(ys))

        #smooth the curve
        smoothedYs = []
        window = []
        for y in ys:
            window.append(y)
            if len(window)>windowSize:
                window.pop(0)
            smoothedYs.append(sum(window)/windowSize)
        plt.plot(x,smoothedYs, label=label + "(" + agents[i] + ")")
        plt.title(environment)
        plt.xlabel("episodes")
        plt.ylabel("reward (smoothed)")
        plt.legend(loc=(0,-0.45))

    #plot metrics
plot(metrics, 0, "reward")
#plt.show()
plt.savefig(f"charts/{environment}_episodes{nEpisodes}_winsize{windowSize}_{datetime.datetime.now().strftime('%Y.%m.%d')}.png", bbox_inches="tight")