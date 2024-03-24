import matplotlib.pyplot as plt
import numpy as np

npz = np.load("metrics/metrics_2024.03.23-23.52.25.npz")
environment = npz["environments"][0]
agents = npz["agents"]
nEpisodes = npz["nEpisodes"][0]
metrics = npz["data"]

def plot(yss, j, label):
    match environment:
        case "MazeEnv":
            plt.axhline(y=4767/2, color="grey")
        case "TagEnv":
            plt.axhline(y=565/2, color="grey")
        case "TTTEnv":
            plt.axhline(y=1000/2, color="lightgrey")
    for i in range(len(agents)):
        ys = yss[i][j]
        x = range(len(ys))

        #smooth the curve
        smoothedYs = []
        window = []
        windowSize = nEpisodes/10
        for y in ys:
            window.append(y)
            if len(window)>windowSize:
                window.pop(0)
            smoothedYs.append(sum(window)/windowSize)
        plt.plot(x,smoothedYs, label=label + "(" + agents[i] + ")")
        plt.title(environment)
        plt.legend()

    #plot metrics
plot(metrics, 0, "reward")
plt.show()