"""
Code based on https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

Policy/target explained: https://ai.stackexchange.com/questions/21485/how-and-when-should-we-update-the-q-target-in-deep-q-learning
Exploding gradients: https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem
"""
import copy
import os
import random
from collections import namedtuple, deque
from typing import TypedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import settings as s
import events as e

MOVED_TOWARD_COIN = "MOVED_TOWARD_COIN"
DID_NOT_MOVE_TOWARD_COIN = "DID_NOT_MOVE_TOWARD_COIN"
MOVED_TOWARD_CRATE = "MOVED_TOWARD_CRATE"
DID_NOT_MOVE_TOWARD_CRATE = "DID_NOT_MOVE_TOWARD_CRATE"
MOVED_TOWARD_SAFETY = "MOVED_TOWARD_SAFETY"
DID_NOT_MOVE_TOWARD_SAFETY = "DID_NOT_MOVE_TOWARD_SAFETY"
PLACED_USEFUL_BOMB = "PLACED_USEFUL_BOMB"
DID_NOT_PLACE_USEFUL_BOMB = "DID_NOT_PLACE_USEFUL_BOMB"

cwd = os.path.abspath(os.path.dirname(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

BATCH_SIZE = 128  # number of transitions sampled from replay buffer
GAMMA = 0.99  # discount factor (for rewards in future states)
EPS_START = 0.9  # starting value of epsilon (for taking random actions)
EPS_END = 0.1  # ending value of epsilon
EPS_DECAY = 10  # how many steps until full epsilon decay
TAU = 0.05  # update rate of the target network
LR = 1e-4  # learning rate of the optimizer
OPTIMIZER = optim.AdamW  # the optimizer

POLICY_MODEL_PATH = f"{cwd}/policy-model.pt"
TARGET_MODEL_PATH = f"{cwd}/target-model.pt"

FEATURE_SIZE = 21  # how many features our model has; ugly but hard to not hardcode

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


# TODO ended here: the search functions MUST take into account that we can't walk somewhere where we'll die
#  implement this via doing next_states on the gamestate and searching the space that way
#  will be slower but more precise


class Game(TypedDict):
    """For typehints."""
    round: int
    step: int
    field: np.ndarray
    bombs: list[tuple[tuple[int, int], int]]
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


def _is_alive(game_state: Game) -> bool:
    return game_state is not None


def _tile_is_free(game_state: Game, x: int, y: int) -> bool:
    for obstacle in [p for (p, _) in game_state['bombs']] + [p for (_, _, _, p) in game_state['others']]:
        if obstacle == (x, y):
            return False

    return game_state['field'][x][y] == 0 and game_state['explosion_map'][x][y] == 0


def _get_blast_coords(game_state: Game, x: int, y: int):
    blast_coords = [(x, y)]
    field = game_state['field']

    for i in range(1, s.BOMB_POWER + 1):
        if field[x + i][y] == -1:
            break
        blast_coords.append((x + i, y))
    for i in range(1, s.BOMB_POWER + 1):
        if field[x - i][y] == -1:
            break
        blast_coords.append((x - i, y))
    for i in range(1, s.BOMB_POWER + 1):
        if field[x][y + i] == -1:
            break
        blast_coords.append((x, y + i))
    for i in range(1, s.BOMB_POWER + 1):
        if field[x][y - i] == -1:
            break
        blast_coords.append((x, y - i))

    return blast_coords


def _next_game_state(game_state: Game, action: str) -> Game | None:
    """Return a new game state by progressing the current one given the action.
    Assumes that all other players stand perfectly still."""
    game_state = copy.deepcopy(game_state)

    # 1. self.poll_and_run_agents() - only us move
    (name, score, bomb, (x, y)) = game_state['self']
    if action == 'UP' and _tile_is_free(game_state, x, y - 1):
        y -= 1
    elif action == 'DOWN' and _tile_is_free(game_state, x, y + 1):
        y += 1
    elif action == 'LEFT' and _tile_is_free(game_state, x - 1, y):
        x -= 1
    elif action == 'RIGHT' and _tile_is_free(game_state, x + 1, y):
        x += 1
    elif action == 'BOMB' and game_state['self'][2]:
        game_state['bombs'].append(((x, y), s.BOMB_TIMER))
    elif action == 'WAIT':
        pass
    else:
        return None

    game_state['self'] = (name, score, bomb, (x, y))

    # 2. self.collect_coins() - not important for now

    # 3. self.update_explosions()
    game_state["explosion_map"] = np.clip(game_state["explosion_map"] - 1, 0, np.inf)

    # 4. self.update_bombs()
    i = 0
    while i < len(game_state['bombs']):
        ((x, y), t) = game_state['bombs'][i]
        t -= 1

        if t <= 0:
            game_state['bombs'].pop(i)

            blast_coords = _get_blast_coords(game_state, x, y)

            for (x, y) in blast_coords:
                game_state['field'][x][y] = 0
                game_state["explosion_map"][x][y] = s.EXPLOSION_TIMER
        else:
            game_state['bombs'][i] = ((x, y), t)
            i += 1

    # 5. self.evaluate_explosions() - kill agents
    x, y = game_state['self'][3]
    if game_state["explosion_map"][x][y] != 0:
        return None  # we died

    # TODO: kill others

    return game_state


def _can_escape_after_placement(game_state: Game) -> bool:
    game_state = copy.deepcopy(game_state)

    x, y = game_state['self'][3]
    game_state['bombs'].append(((x, y), s.BOMB_TIMER - 1))

    return _directions_to_safety(game_state) != 0


def _directions_to_coins(game_state: Game) -> list[int]:
    start = game_state["self"][-1]
    queue = deque([(game_state, 0)])
    explored = {start: None}

    candidates = set([])
    candidate_distance = None

    while len(queue) != 0:
        current_game_state, distance = queue.popleft()
        current = current_game_state['self'][-1]

        if len(candidates) != 0 and candidate_distance < distance:
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


def _directions_to_crates(game_state: Game) -> list[int]:
    start = game_state["self"][-1]
    queue = deque([(game_state, 0)])
    explored = {start: None}

    candidates = set([])
    candidate_distance = None

    while len(queue) != 0:
        current_game_state, distance = queue.popleft()
        current = current_game_state['self'][-1]

        if len(candidates) != 0 and candidate_distance < distance:
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


# def _directions_to_crates(game_state: Game) -> list[int]:
#    # TODO: I'm copy pasted!
#    start = game_state["self"][-1]
#    queue = deque([(start, 0)])
#    explored = {start: None}
#
#    candidates = set([])
#    candidate_distance = None
#
#    while len(queue) != 0:
#        current, distance = queue.popleft()
#
#        if len(candidates) != 0 and candidate_distance < distance:
#            break
#
#        if game_state["field"][current[0]][current[1]] == 1:
#            if explored[current] == start:
#                return [4]
#
#            while explored[current] != start:
#                current = explored[current]
#
#            candidates.add(DELTAS.index((current[0] - start[0], current[1] - start[1])))
#            candidate_distance = distance
#            continue
#
#        for dx, dy in DELTAS:
#            neighbor = (current[0] + dx, current[1] + dy)
#
#            if neighbor in explored:
#                continue
#
#            explored[neighbor] = current
#
#            if game_state["field"][neighbor[0]][neighbor[1]] in [0, 1]:
#                queue.append((neighbor, distance + 1))
#
#    return list(candidates)


def _get_validity_vector(game_state: Game) -> list[int]:
    """Return a vector of valid directions from the given state (i.e. if you walk here you die instantly)."""
    return [
        1 if _is_alive(_next_game_state(game_state, action)) else 0
        for action in ACTIONS
    ]


def _is_in_danger(game_state):
    x, y = game_state['self'][3]
    for ((bx, by), _) in game_state['bombs']:
        if (x, y) in _get_blast_coords(game_state, bx, by):
            return True
    return False


def _directions_to_safety(game_state) -> list[int]:
    if not _is_in_danger(game_state):
        return []

    x, y = game_state['self'][3]
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


def state_to_features(game_state: Game | None) -> torch.Tensor | None:
    """
    # 0..4 - direction to closest coin
    # 5..9 - direction to closest crate
    # 10..14 - direction to where placing a bomb will hurt another player  # TODO
    # 15..19 - direction to safety; has a one only if is in danger
    # 20 - can we place a bomb (and live to tell the tale)?
    """
    if game_state is None:
        return None

    feature_vector = [0] * (5 + 5 + 5 + 5 + 1)

    if v := _directions_to_coins(game_state):
        for i in v:
            feature_vector[i] = 1

    if v := _directions_to_crates(game_state):
        for i in v:
            feature_vector[i + 5] = 1

    if v := _directions_to_safety(game_state):
        for i in v:
            feature_vector[i + 15] = 1

        # if we need to run, mask other features
        for i in range(3):
            for j in v:
                feature_vector[j + 5 * i] &= feature_vector[j + 15]

    if game_state["self"][2] and _can_escape_after_placement(game_state):
        feature_vector[20] = 1

    return torch.tensor([feature_vector], device=device, dtype=torch.float)


def _is_bomb_useful(game_state):
    x, y = game_state['self'][3]
    for bx, by in _get_blast_coords(game_state, x, y):
        if game_state['field'][bx][by] == 1:
            return True

        if (bx, by) in [a[3] for a in game_state['others']]:
            return True
    return False


def _reward_from_events(self, events: list[str]) -> torch.Tensor:
    game_rewards = {
        # hunt coins
        MOVED_TOWARD_COIN: 10,
        DID_NOT_MOVE_TOWARD_COIN: -15,
        e.COIN_COLLECTED: 50,
        # blow up crates
        MOVED_TOWARD_CRATE: 1,
        # basic stuff
        e.KILLED_OPPONENT: 500,
        e.KILLED_SELF: -1000,
        e.GOT_KILLED: -1000,
        e.INVALID_ACTION: -10,
        MOVED_TOWARD_SAFETY: 100,
        DID_NOT_MOVE_TOWARD_SAFETY: -100,
        # be active!
        e.WAITED: -5,
        # meaningful bombs
        PLACED_USEFUL_BOMB: 20,
        DID_NOT_PLACE_USEFUL_BOMB: -1000,
        e.CRATE_DESTROYED: 10,
        e.COIN_FOUND: 10,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

    return torch.tensor([reward_sum], device=device, dtype=torch.float)


def _process_game_event(self, old_game_state: Game, self_action: str,
                        new_game_state: Game | None, events: list[str]):
    state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)
    action = torch.tensor([[ACTIONS.index(self_action)]], dtype=torch.long)

    state_list = state.tolist()[0]

    # 0..4 - direction to closest coin
    # 5..9 - direction to closest crate
    # 10..14 - direction to where placing a bomb will hurt another player  # TODO
    # 15..19 - direction to safety; has a one only if is in danger

    # don't add more events to an invalid action... it's literally just inval
    if not e.INVALID_ACTION in events:
        moving_events = [
            (MOVED_TOWARD_COIN, DID_NOT_MOVE_TOWARD_COIN, 0, 5),
            (MOVED_TOWARD_CRATE, DID_NOT_MOVE_TOWARD_CRATE, 5, 10),
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

        if self_action == "BOMB":
            if _is_bomb_useful(old_game_state):
                events.append(PLACED_USEFUL_BOMB)
            else:
                events.append(DID_NOT_PLACE_USEFUL_BOMB)

        # TODO: +event: moving towards the center

    self.logger.info(f"State vector: {state_list}")
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
    self.policy_model = DQN(FEATURE_SIZE, len(ACTIONS)).to(device)
    self.target_model = DQN(FEATURE_SIZE, len(ACTIONS)).to(device)
    self.target_model.load_state_dict(self.policy_model.state_dict())

    self.model = self.policy_model

    self.optimizer = OPTIMIZER(self.policy_model.parameters(), lr=LR, amsgrad=True)
    self.memory = ReplayMemory(1000)

    self.max_score = 0


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

    _process_game_event(self, last_game_state, last_action, None, events)

    torch.save(self.policy_model, POLICY_MODEL_PATH)
    torch.save(self.target_model, TARGET_MODEL_PATH)

    if last_game_state['self'][1] > self.max_score:
        print(last_game_state['self'][1])
        self.max_score = last_game_state['self'][1]
