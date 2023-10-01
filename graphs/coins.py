import matplotlib.pyplot as plt
import numpy as np


FILE = "../agent_code/binary_agent/models/plot.txt.0"

with open(FILE) as f:
    x, score, reward, steps = f.read().splitlines()

    x = np.array(list(map(int, x.split())))
    score = np.array(list(map(int, score.split()))) / 50
    reward = np.array(list(map(float, reward.split()))) / 20000
    steps = np.array(list(map(int, steps.split()))) / 400


fig = plt.figure(figsize=(12, 6))
ax = plt.axes()

plot_score, = ax.plot(x, score, '-', color='blue', label='normalized game score')
plot_reward, = ax.plot(x, reward, color='red', label='reward / 20000', linestyle='dashed', linewidth=1)
plot_steps, = ax.plot(x, steps, '-', color='green', label='normalized steps')
ax.legend(loc='lower right')
ax.set(ylim=(-0.25, 1.1))

plt.tight_layout()
plt.savefig("coins.pdf")
