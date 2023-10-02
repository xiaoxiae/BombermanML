# BombermanML
Implementations of various ML algorithms for playing the game of bomberman.

## Contents
The repository contains (in addition to the original repository) the following files:

```
.
├── agent_code
│   :   # binary agents (features are binary vectors)
│   ├── binary_agent
│   ├── binary_agent_v1
│   ├── binary_agent_v2
│   ├── binary_agent_v3
│   ├── binary_agent_v4
│   ├── binary_agent_v5
│   ├── binary_agent_v6
│   :   # binary distance agents
│   :   # same as binary but values are 1/distance
│   ├── binary_distance_agent_v1
│   ├── binary_distance_agent_v2
│   :   # distance agent (features are distances)
│   :   # differs from binary_distance (gives all distances, not just shortest)
│   ├── distance_agent
│   ├── distance_agent_v1
│   :   # q-table agents
│   ├── q_agent
│   ├── q_agent_v1
│   ├── q_agent_v2
│   └── q_agent_v3
├── graphs      # matplotlib graphs for the agent report
├── elo         # elo data
├── elo.py      # script for calculating elo
├── train.py    # script for training agents
└── train_q.py  # old script for training q-table
```
