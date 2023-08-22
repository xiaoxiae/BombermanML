"""
Code based on https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

Policy/target explained: https://ai.stackexchange.com/questions/21485/how-and-when-should-we-update-the-q-target-in-deep-q-learning
Exploding gradients: https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem
"""
import os
import random
from collections import namedtuple, deque
from typing import TypedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import events as e

MOVED_TOWARD_COIN = "MOVED_TOWARD_COIN"

cwd = os.path.abspath(os.path.dirname(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

BATCH_SIZE = 5  # number of transitions sampled from replay buffer
GAMMA = 0.5  # discount factor (for rewards in future states)
EPS_START = 0.9  # starting value of epsilon (for taking random actions)
EPS_END = 0.01  # ending value of epsilon
EPS_DECAY = 5  # how many steps until full epsilon decay
TAU = 0.05  # update rate of the target network
LR = 1e-4  # learning rate of the optimizer
OPTIMIZER = optim.AdamW  # the optimizer

POLICY_MODEL_PATH = f"{cwd}/policy-model.pt"
TARGET_MODEL_PATH = f"{cwd}/target-model.pt"

FEATURE_SIZE = 5  # how many features our model has; ugly but hard to not hardcode

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class GameState(TypedDict):
    """For typehints."""
    round: int
    step: int
    field: np.ndarray
    bombs: list[tuple[int, int], int]
    explosion_map: np.ndarray
    coins: list[tuple[int, int]]
    self: tuple[str, int, bool, tuple[int, int]]
    others: list[tuple[str, int, bool, tuple[int, int]]]
    user_input: str | None


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations: int, n_actions: int, hidden_size: int = 128):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, n_actions)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def _optimize_model(self):
    if len(self.memory) < BATCH_SIZE:
        return

    transitions = self.memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                  dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = self.policy_model(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(self.policy_model.parameters(), 100)
    self.optimizer.step()


def _direction_to_coin(game_state: GameState) -> int:
    start = game_state["self"][-1]
    queue = deque([start])
    explored = {start: None}

    while len(queue) != 0:
        current = queue.popleft()

        if current in game_state["coins"]:
            if current == start:
                return 4

            # backtrack and find where we initially went to get here
            while explored[current] is not start:
                current = explored[current]

            return DELTAS.index((current[0] - start[0], current[1] - start[1]))

        for dx, dy in DELTAS:
            neighbor = (current[0] + dx, current[1] + dy)

            if neighbor in explored:
                continue

            explored[neighbor] = current

            if game_state["field"][neighbor[0]][neighbor[1]] == 0:
                queue.append(neighbor)


def _state_to_features(game_state: GameState | None) -> torch.Tensor | None:
    if game_state is None:
        return None

    feature_vector = [0, 0, 0, 0, 0]

    # first 4 features are boolean -- direction to the closest coin
    coin_action_index = _direction_to_coin(game_state)
    if coin_action_index is not None:
        feature_vector[coin_action_index] = 1

    return torch.tensor([feature_vector], device=device, dtype=torch.float)


def _reward_from_events(self, events: list[str]) -> torch.Tensor:
    game_rewards = {
        e.BOMB_DROPPED: 3,
        e.CRATE_DESTROYED: 5,
        e.COIN_FOUND: 1,
        MOVED_TOWARD_COIN: 1,
        e.COIN_COLLECTED: 5,
        e.KILLED_OPPONENT: 100,
        e.KILLED_SELF: -1000,
        e.GOT_KILLED: -1000,
        e.INVALID_ACTION: -1000,
        e.SURVIVED_ROUND: 1000,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

    return torch.tensor([reward_sum], device=device, dtype=torch.float)


def setup_training(self):
    self.policy_model = DQN(FEATURE_SIZE, len(ACTIONS)).to(device)
    self.target_model = DQN(FEATURE_SIZE, len(ACTIONS)).to(device)
    self.target_model.load_state_dict(self.policy_model.state_dict())

    self.model = self.policy_model

    self.optimizer = OPTIMIZER(self.policy_model.parameters(), lr=LR, amsgrad=True)
    self.memory = ReplayMemory(1000)


def game_events_occurred(self, old_game_state: GameState, self_action: str,
                         new_game_state: GameState, events: list[str]):
    """
    Called once per step to allow intermediate rewards based on game events.
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    state = _state_to_features(old_game_state)
    new_state = _state_to_features(new_game_state)
    action = torch.tensor([[ACTIONS.index(self_action)]], dtype=torch.long)

    state_list = state.tolist()[0]
    for i in range(5):
        if np.isclose(state_list[i], 1) and self_action == ACTIONS[i]:
            events.append(MOVED_TOWARD_COIN)

    reward = _reward_from_events(self, events)

    self.memory.push(state, action, new_state, reward)

    # Optimize
    _optimize_model(self)

    # Soft-update
    target_net_state_dict = self.target_model.state_dict()
    policy_net_state_dict = self.policy_model.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
    self.target_model.load_state_dict(target_net_state_dict)


def end_of_round(self, last_game_state: dict, last_action: str, events: list[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.
    """
    # TODO: implement some stuff here?

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    torch.save(self.policy_model, POLICY_MODEL_PATH)
    torch.save(self.target_model, TARGET_MODEL_PATH)

    # TODO: store the model
