import matplotlib.pyplot as plt
import numpy as np

stats = [
    ["binary_agent_v1", np.array((624, 90, 286), dtype=float) / 1000],
    ["binary_agent_v4", np.array((454, 134, 412), dtype=float) / 1000],
    ["coin_collector_agent", np.array((878, 4, 118), dtype=float) / 1000],
    ["distance_agent_v1", np.array((474, 139, 387), dtype=float) / 1000],
    ["peaceful_agent", np.array((1000, 0, 0), dtype=float) / 1000],
    ["q_agent_v1", np.array((100, 0, 0), dtype=float) / 100],
    ["q_agent_v2", np.array((99, 0, 1), dtype=float) / 100],
    ["q_agent_v3", np.array((86, 1, 13), dtype=float) / 100],
    ["random_agent", np.array((1000, 0, 0), dtype=float) / 1000],
    ["rule_based_agent", np.array((837, 9, 154), dtype=float) / 1000],
]

agents = [a[0] for a in stats]
weights = (
    ("Wins", np.array([a[1][0] for a in stats])),
    ("Draws", np.array([a[1][1] for a in stats])),
    ("Losses", np.array([a[1][2] for a in stats])),
)

fig = plt.figure(figsize=(12, 6))
ax = plt.axes()
bottom = np.zeros(len(agents))

width = 0.7

for (boolean, weight), color in zip(weights, ["Green", "Orange", "Red"]):
    ax.bar(agents, weight, width, label=boolean, bottom=bottom, color=color)
    bottom += weight

plt.xticks(rotation=25)

ax.set_ylabel('Win / Draw / Loss percentage')
plt.tight_layout()
plt.savefig("stacked.pdf")
plt.show()
