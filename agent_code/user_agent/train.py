"""
Code based on https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

Policy/target explained: https://ai.stackexchange.com/questions/21485/how-and-when-should-we-update-the-q-target-in-deep-q-learning
Exploding gradients: https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem
"""
import copy
import os
import random
from collections import namedtuple, deque
from functools import lru_cache, cache
from typing import TypedDict
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import events as e
import settings as s

MOVED_TOWARD_COIN = "MOVED_TOWARD_COIN"
DID_NOT_MOVE_TOWARD_COIN = "DID_NOT_MOVE_TOWARD_COIN"
MOVED_TOWARD_CRATE = "MOVED_TOWARD_CRATE"
DID_NOT_MOVE_TOWARD_CRATE = "DID_NOT_MOVE_TOWARD_CRATE"
MOVED_TOWARD_SAFETY = "MOVED_TOWARD_SAFETY"
DID_NOT_MOVE_TOWARD_SAFETY = "DID_NOT_MOVE_TOWARD_SAFETY"
PLACED_USEFUL_BOMB = "PLACED_USEFUL_BOMB"
PLACED_SUPER_USEFUL_BOMB = "PLACED_SUPER_USEFUL_BOMB"
DID_NOT_PLACE_USEFUL_BOMB = "DID_NOT_PLACE_USEFUL_BOMB"
MOVED_TOWARD_PLAYER = "MOVED_TOWARD_PLAYER"
DID_NOT_MOVE_TOWARD_PLAYER = "DID_NOT_MOVE_TOWARD_PLAYER"
USELESS_WAIT = "USELESS_WAIT"

cwd = os.path.abspath(os.path.dirname(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

BATCH_SIZE = 128  # number of transitions sampled from replay buffer
MEMORY_SIZE = 5000  # number of transitions to keep in the replay buffer
GAMMA = 0.99  # discount factor (for rewards in future states)
EPS_START = 0.1  # starting value of epsilon (for taking random actions)
EPS_END = 0.05  # ending value of epsilon
EPS_DECAY = 10  # how many steps until full epsilon decay
TAU = 1e-3  # update rate of the target network
LR = 1e-4  # learning rate of the optimizer
OPTIMIZER = optim.Adam  # the optimizer

EMPTY_FIELD = np.array([
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
    [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
    [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
    [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
    [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
    [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
    [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
    [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
])

POLICY_MODEL_PATH = f"{cwd}/policy-model.pt"
TARGET_MODEL_PATH = f"{cwd}/target-model.pt"

FEATURE_VECTOR_SIZE = 21  # how many features our model has; ugly but hard to not hardcode

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class Game(TypedDict):
    """For typehints."""
    field: np.ndarray
    bombs: list[tuple[tuple[int, int], int]]
    explosion_map: np.ndarray
    coins: list[tuple[int, int]]
    self: tuple[str, int, bool, tuple[int, int]]
    others: list[tuple[str, int, bool, tuple[int, int]]]
    # these aren't really important, so we don't expect them to be there
    # round: int
    # step: int
    # user_input: str | None


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

    def __init__(self, n_observations: int, n_actions: int, hidden_size: int = 1024, hidden_count: int = 1):
        super(DQN, self).__init__()

        self.layers = nn.ModuleList(
            [nn.Linear(n_observations, hidden_size)] \
            + [nn.Linear(hidden_size, hidden_size) for _ in range(hidden_count)] \
            + [nn.Linear(hidden_size, n_actions)]
        )

    def forward(self, x: torch.Tensor):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


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


def _is_alive(game_state: Game) -> bool:
    return game_state is not None


def _tile_is_free(game_state: Game, x: int, y: int) -> bool:
    for obstacle in [p for (p, _) in game_state['bombs']] + [p for (_, _, _, p) in game_state['others']]:
        if obstacle == (x, y):
            return False

    return game_state['field'][x][y] == 0 and game_state['explosion_map'][x][y] == 0


@cache
def _get_blast_coords(x: int, y: int) -> tuple[tuple[int, int]]:
    if EMPTY_FIELD[x][y] == -1:
        return tuple()

    blast_coords = [(x, y)]

    for i in range(1, s.BOMB_POWER + 1):
        if EMPTY_FIELD[x + i][y] == -1:
            break
        blast_coords.append((x + i, y))
    for i in range(1, s.BOMB_POWER + 1):
        if EMPTY_FIELD[x - i][y] == -1:
            break
        blast_coords.append((x - i, y))
    for i in range(1, s.BOMB_POWER + 1):
        if EMPTY_FIELD[x][y + i] == -1:
            break
        blast_coords.append((x, y + i))
    for i in range(1, s.BOMB_POWER + 1):
        if EMPTY_FIELD[x][y - i] == -1:
            break
        blast_coords.append((x, y - i))

    return tuple(blast_coords)


def _next_game_state(game_state: Game, action: str) -> Game | None:
    """Return a new game state by progressing the current one given the action.
    Assumes that all other players stand perfectly still.
    If the action is invalid, returns None."""
    game_state = copy.copy(game_state)
    game_state['bombs'] = list(game_state['bombs'])

    # 1. self.poll_and_run_agents() - only us move
    (name, score, bomb, (x, y)) = game_state['self']
    if action == 'UP':
        if _tile_is_free(game_state, x, y - 1):
            y -= 1
        else:
            return None
    elif action == 'DOWN':
        if _tile_is_free(game_state, x, y + 1):
            y += 1
        else:
            return None
    elif action == 'LEFT':
        if _tile_is_free(game_state, x - 1, y):
            x -= 1
        else:
            return None
    elif action == 'RIGHT':
        if _tile_is_free(game_state, x + 1, y):
            x += 1
        else:
            return None
    elif action == 'BOMB':
        if game_state['self'][2]:
            game_state['bombs'].append(((x, y), s.BOMB_TIMER))
        else:
            return None
    elif action == 'WAIT':
        pass
    else:
        return None

    game_state['self'] = (name, score, bomb, (x, y))

    # 2. self.collect_coins() - not important for now

    # 3. self.update_explosions()
    game_state["explosion_map"] = np.clip(game_state["explosion_map"] - 1, 0, None)

    # 4. self.update_bombs()
    game_state['field'] = np.array(game_state['field'])
    i = 0
    while i < len(game_state['bombs']):
        ((x, y), t) = game_state['bombs'][i]
        t -= 1

        if t < 0:
            game_state['bombs'].pop(i)
            blast_coords = _get_blast_coords(x, y)

            for (x, y) in blast_coords:
                game_state['field'][x][y] = 0
                game_state["explosion_map"][x][y] = s.EXPLOSION_TIMER
        else:
            game_state['bombs'][i] = ((x, y), t)
            i += 1

    # 5. self.evaluate_explosions() - kill agents
    # self
    x, y = game_state['self'][3]
    if game_state["explosion_map"][x][y] != 0:
        return None  # we died

    # others
    if len(game_state['others']) != 0:
        i = 0
        game_state['others'] = list(game_state['others'])
        while i < len(game_state['others']):
            x, y = game_state['others'][i][-1]
            if game_state["explosion_map"][x][y] != 0:
                game_state['others'].pop(i)
            else:
                i += 1

    return game_state


def _can_escape_after_placement(game_state: Game) -> bool:
    game_state = copy.copy(game_state)

    x, y = game_state['self'][3]
    game_state['bombs'] = list(game_state['bombs']) + [((x, y), s.BOMB_TIMER)]

    return len(_directions_to_safety(game_state)) != 0


def _directions_to_coins(game_state: Game) -> list[int]:
    # no coins
    if len(game_state['coins']) == 0:
        return []

    start = game_state["self"][-1]
    queue = deque([(game_state, 0)])
    explored = {start: None}

    candidates = set([])
    candidate_distance = None

    while len(queue) != 0:
        current_game_state, distance = queue.popleft()
        current = current_game_state['self'][-1]

        if candidate_distance is not None and candidate_distance < distance:
            break

        if current in current_game_state["coins"]:
            if current == start:
                return [4]

            while explored[current] != start:
                current = explored[current]

            candidates.add(DELTAS.index((current[0] - start[0], current[1] - start[1])))
            candidate_distance = distance
            continue

        for (dx, dy), action in zip(DELTAS, ACTIONS):
            neighbor = (current[0] + dx, current[1] + dy)

            if neighbor in explored:
                continue

            explored[neighbor] = current

            if _tile_is_free(current_game_state, *neighbor):
                new_game_state = _next_game_state(current_game_state, action)

                if new_game_state is None:
                    continue

                queue.append((new_game_state, distance + 1))

    return list(candidates)


def _directions_to_enemy(game_state: Game):
    if len(game_state['others']) == 0:
        return []

    start = game_state["self"][-1]
    queue = deque([(game_state, 0)])
    explored = {start: None}

    candidates = set([])
    candidate_distance = None

    while len(queue) != 0:
        current_game_state, distance = queue.popleft()
        current = current_game_state['self'][-1]

        if candidate_distance is not None and candidate_distance < distance:
            break

        for (dx, dy), action in zip(DELTAS, ACTIONS):
            neighbor = (current[0] + dx, current[1] + dy)

            if neighbor in explored:
                continue

            explored[neighbor] = current

            for n in current_game_state['others']:
                if n[-1] in _get_blast_coords(*neighbor):
                    if current == start:
                        return [4]

                    while explored[current] != start:
                        current = explored[current]

                    candidates.add(DELTAS.index((current[0] - start[0], current[1] - start[1])))
                    candidate_distance = distance

                    break
            else:
                if _tile_is_free(current_game_state, *neighbor):
                    new_game_state = _next_game_state(current_game_state, action)

                    if new_game_state is None:
                        continue

                    queue.append((new_game_state, distance + 1))

    return list(candidates)


def _directions_to_crates(game_state: Game) -> list[int]:
    # no crates
    if 1 not in game_state['field']:
        return []

    start = game_state["self"][-1]
    queue = deque([(game_state, 0)])
    explored = {start: None}

    candidates = set([])
    candidate_distance = None

    while len(queue) != 0:
        current_game_state, distance = queue.popleft()
        current = current_game_state['self'][-1]

        if candidate_distance is not None and candidate_distance < distance:
            break

        for (dx, dy), action in zip(DELTAS, ACTIONS):
            neighbor = (current[0] + dx, current[1] + dy)

            if neighbor in explored:
                continue

            explored[neighbor] = current

            if _tile_is_free(current_game_state, *neighbor):
                new_game_state = _next_game_state(current_game_state, action)

                if new_game_state is None:
                    continue

                queue.append((new_game_state, distance + 1))
            elif current_game_state['field'][neighbor[0]][neighbor[1]] == 1:
                if current == start:
                    return [4]

                while explored[current] != start:
                    current = explored[current]

                candidates.add(DELTAS.index((current[0] - start[0], current[1] - start[1])))
                candidate_distance = distance
                continue

    return list(candidates)


def _is_in_danger(game_state):
    x, y = game_state['self'][3]
    for ((bx, by), _) in game_state['bombs']:
        if (x, y) in _get_blast_coords(bx, by):
            return True
    return False


def _directions_to_safety(game_state) -> list[int]:
    if not _is_in_danger(game_state):
        return []

    queue = deque([(game_state, [])])

    valid_actions = set()

    while len(queue) != 0:
        current_game_state, action_history = queue.popleft()

        if not _is_in_danger(current_game_state):
            valid_actions.add(action_history[0])
            continue

        for action in ACTIONS[:5]:
            new_game_state = _next_game_state(current_game_state, action)

            if new_game_state is None:
                continue

            queue.append((new_game_state, list(action_history) + [action]))

    return [ACTIONS.index(action) for action in valid_actions]


@lru_cache(maxsize=10000)
def _state_to_features(game_state: tuple | None) -> torch.Tensor | None:
    """
    # 0..4 - direction to closest coin
    # 5..9 - direction to closest crate
    # 10..14 - direction to where placing a bomb will hurt another player
    # 15..19 - direction to safety; has a one only if is in danger
    # 20 - can we place a bomb (and live to tell the tale)?
    """
    game_state: Game = {
        'field': np.array(game_state[0]),
        'bombs': list(game_state[1]),
        'explosion_map': np.array(game_state[2]),
        'coins': list(game_state[3]),
        'self': game_state[4],
        'others': list(game_state[5]),
    }

    feature_vector = [0] * FEATURE_VECTOR_SIZE

    if v := _directions_to_coins(game_state):
        for i in v:
            feature_vector[i] = 1

    if v := _directions_to_crates(game_state):
        for i in v:
            feature_vector[i + 5] = 1

    if v := _directions_to_enemy(game_state):
        for i in v:
            feature_vector[i + 10] = 1

    if v := _directions_to_safety(game_state):
        for i in v:
            feature_vector[i + 15] = 1

        # if we can get to safety by something other than waiting, don't wait
        if v != [4]:
            feature_vector[19] = 0

        # if we need to run away, mask other features to do that too
        for i in range(3):
            for j in range(5):
                feature_vector[j + 5 * i] &= feature_vector[j + 15]
    else:
        # TODO: directions away from the bomb
        #  if an enemy is blocking the way, it's still better to just run away from the bomb
        #  the code is probably the same but we ignore all player positions
        pass

    if game_state["self"][2] and _can_escape_after_placement(game_state):
        feature_vector[20] = 1

    return torch.tensor([feature_vector], device=device, dtype=torch.float)


def state_to_features(game_state: Game | None) -> torch.Tensor | None:
    if game_state is None:
        return None

    return _state_to_features(
        (
            tuple(tuple(r) for r in game_state['field']),
            tuple(game_state['bombs']),
            tuple(tuple(r) for r in game_state['explosion_map']),
            tuple(game_state['coins']),
            game_state['self'],
            tuple(game_state['others']),
        )
    )


def _is_bomb_useful(game_state):
    x, y = game_state['self'][3]
    for bx, by in _get_blast_coords(x, y):
        # destroys crate
        if game_state['field'][bx][by] == 1:
            return True

        # kills a player
        if (bx, by) in [a[-1] for a in game_state['others']]:
            return True

    return False


def _reward_from_events(self, events: list[str]) -> torch.Tensor:
    game_rewards = {
        # hunt coins
        MOVED_TOWARD_COIN: 10,
        DID_NOT_MOVE_TOWARD_COIN: -100,
        e.COIN_COLLECTED: 100,
        # hunt people
        e.KILLED_OPPONENT: 500,
        MOVED_TOWARD_PLAYER: 10,
        DID_NOT_MOVE_TOWARD_PLAYER: -10,
        # blow up crates
        MOVED_TOWARD_CRATE: 5,
        DID_NOT_MOVE_TOWARD_CRATE: -5,
        # basic stuff
        e.GOT_KILLED: -1000,
        e.KILLED_SELF: -1000,
        e.SURVIVED_ROUND: 1000,
        e.INVALID_ACTION: -10,
        MOVED_TOWARD_SAFETY: 100,
        DID_NOT_MOVE_TOWARD_SAFETY: -1000,
        # be active!
        USELESS_WAIT: -100,
        # meaningful bombs
        PLACED_USEFUL_BOMB: 50,
        PLACED_SUPER_USEFUL_BOMB: 100,
        DID_NOT_PLACE_USEFUL_BOMB: -1000,
        e.CRATE_DESTROYED: 10,
        e.COIN_FOUND: 10,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    self.logger.debug(f"Awarded {reward_sum} for events {', '.join(events)}")

    if self.manual:
        print(f"Awarded {reward_sum} for events {', '.join(events)}")

    return torch.tensor([reward_sum], device=device, dtype=torch.float)


def _process_game_event(self, old_game_state: Game, self_action: str,
                        new_game_state: Game | None, events: list[str]):
    state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)
    action = torch.tensor([[ACTIONS.index(self_action)]], device=device, dtype=torch.long)

    state_list = state.tolist()[0]

    # don't add more events to an invalid action, because it's just invalid
    moving_events = [
        (MOVED_TOWARD_COIN, DID_NOT_MOVE_TOWARD_COIN, 0, 5),
        (MOVED_TOWARD_CRATE, DID_NOT_MOVE_TOWARD_CRATE, 5, 10),
        (MOVED_TOWARD_PLAYER, DID_NOT_MOVE_TOWARD_PLAYER, 10, 15),
        (MOVED_TOWARD_SAFETY, DID_NOT_MOVE_TOWARD_SAFETY, 15, 20),
    ]

    for pos_event, neg_event, i, j in moving_events:
        if np.isclose(sum(state_list[i:j]), 0):
            continue

        for i in range(i, j):
            if np.isclose(state_list[i], 1) and self_action == ACTIONS[i % 5]:
                events.append(pos_event)
                break
        else:
            events.append(neg_event)

    if self_action == "BOMB" and old_game_state['self'][2]:
        if _is_bomb_useful(old_game_state) and state_list[20] == 1:
            # if it endangers a player
            if state_list[14]:
                events.append(PLACED_SUPER_USEFUL_BOMB)
            else:
                events.append(PLACED_USEFUL_BOMB)
        else:
            events.append(DID_NOT_PLACE_USEFUL_BOMB)

    # if we wait, make sure it's meaningful (i.e. we weren't recommended to move somewhere)
    if self_action == "WAIT":
        for i in [j + 5 * i for i in range(3) for j in range(4)]:
            if state_list[i] == 1:
                events.append(USELESS_WAIT)
                break
        else:
            # waiting near a crate when we can place a bomb is also useless
            if state_list[20] == 1 and state_list[9] == 1:
                events.append(USELESS_WAIT)

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


def setup_training(self):
    if os.path.exists(POLICY_MODEL_PATH):
        self.policy_model = torch.load(POLICY_MODEL_PATH)
    else:
        self.policy_model = DQN(FEATURE_VECTOR_SIZE, len(ACTIONS)).to(device)

    if os.path.exists(TARGET_MODEL_PATH):
        self.target_model = torch.load(TARGET_MODEL_PATH)
    else:
        self.target_model = DQN(FEATURE_VECTOR_SIZE, len(ACTIONS)).to(device)
        self.target_model.load_state_dict(self.policy_model.state_dict())

    self.model = self.policy_model

    self.optimizer = OPTIMIZER(self.policy_model.parameters(), lr=LR)
    self.memory = ReplayMemory(MEMORY_SIZE)

    self.x = [0]
    self.y = [0]

    self.fig = plt.figure(figsize=(6, 3))
    self.plot, = plt.plot([], [], '-')

    plt.show(block=False)


def game_events_occurred(self, old_game_state: Game, self_action: str, new_game_state: Game, events: list[str]):
    """
    Called once per step to allow intermediate rewards based on game events.
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    _process_game_event(self, old_game_state, self_action, new_game_state, events)


def end_of_round(self, last_game_state: dict, last_action: str, events: list[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    if len(self.x) > 100:
        self.x.pop(0)
        self.y.pop(0)

    self.x.append(self.x[-1] + 1)
    self.y.append(last_game_state['self'][1])

    self.plot.set_data(self.x, self.y)
    self.fig.gca().relim()
    self.fig.gca().autoscale_view()
    self.fig.canvas.draw()
    self.fig.canvas.flush_events()

    _process_game_event(self, last_game_state, last_action, None, events)

    torch.save(self.policy_model, POLICY_MODEL_PATH)
    torch.save(self.target_model, TARGET_MODEL_PATH)
