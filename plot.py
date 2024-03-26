import matplotlib.pyplot as plt
import numpy as np
import datetime
import os

INPUT_PATH = "metrics/final/metrics_MazeEnv_epochs1000_seed0.npz"
OUTPUT_PATH = ""
SAVE_TO_FILE = False
INCLUDE_BASELINE = True

npz = np.load(INPUT_PATH)
environment = npz["environments"][0]
agents = npz["agents"]
nEpisodes = npz["nEpisodes"][0]
metrics = npz["data"]

windowSize = int(nEpisodes/10)

def plot(yss, j, label):
    if INCLUDE_BASELINE:
        baseline = None
        match environment:
            case "MazeEnv":
                baseline = 5580
            case "TagEnv":
                baseline = 1000
            case "TTTEnv":
                baseline = 950
        plt.axhline(y=baseline, color="lightgrey",label="baseline")
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
        plt.xlabel("episode")
        plt.ylabel("reward (smoothed)")
        plt.legend(bbox_to_anchor=(0.6, -0.15))

    #plot metrics
plot(metrics, 0, "reward")

if SAVE_TO_FILE:
    os.makedirs("charts", exist_ok=True)
    plt.savefig(f"charts/{environment}_episodes{nEpisodes}_winsize{windowSize}_{datetime.datetime.now().strftime('%Y.%m.%d')}.png", bbox_inches="tight")
else:
    plt.show()