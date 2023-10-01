import matplotlib.pyplot as plt
import numpy as np


FILE = "../agent_code/binary_agent/models/plot.txt.1"
LIMIT = 100
WINDOW = 5

with open(FILE) as f:
    x, score, reward, steps = f.read().splitlines()

    x = (np.array(list(map(int, x.split()))))[:LIMIT]
    score = (np.array(list(map(int, score.split()))) / 5)[:LIMIT]
    reward = (np.array(list(map(float, reward.split()))) / 10000)[:LIMIT]
    steps = (np.array(list(map(int, steps.split()))) / 400)[:LIMIT]


def moving_average(x):
    return np.convolve(x, np.ones(WINDOW), 'valid') / WINDOW


fig = plt.figure(figsize=(12, 6))
ax = plt.axes()

score = moving_average(score)
reward = moving_average(reward)
steps = moving_average(steps)
x = np.arange(len(score))

plot_score, = ax.plot(x, score, '-', color='blue', label='normalized game score')
plot_reward, = ax.plot(x, reward, color='red', label='reward / 10000', linestyle='dashed', linewidth=1)
plot_steps, = ax.plot(x, steps, '-', color='green', label='normalized steps')
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig("vs-rule.pdf")
plt.show()
