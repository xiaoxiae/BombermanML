import matplotlib.pyplot as plt
import numpy as np

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i] + 5, f"{y[i]:.02f}s", ha = 'center')

fig = plt.figure(figsize=(12, 6))
ax = plt.axes()

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

methods = ["None", "No deepcopy", "No deepcopy + Caching"]
times = [820.945, 67.189, 26.944]

ax.bar(methods, times, label=methods, color=default_colors)

addlabels(methods, times)

ax.set_ylabel('Duration (in s)')

plt.tight_layout()
plt.savefig("optimization.pdf")
plt.show()
