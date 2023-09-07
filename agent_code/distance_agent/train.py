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

# Binarizing means turning the distance vectors into binary vectors indicating
# which distance is the shortest (i.e. what we've been doing before). Used for testing
binarize = False

# Switch between event-based and potential-based rewards.
# Note: the potential-based rewards use hard-coded values for now!
use_potential = True

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

COIN = 'COIN'
CRATE = 'CRATE'
ENEMY = 'ENEMY'
SAFETY = 'SAFETY'

cwd = os.path.abspath(os.path.dirname(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

GAME_REWARDS = {
    # hunt coins
    MOVED_TOWARD_COIN: 2,
    DID_NOT_MOVE_TOWARD_COIN: -6,       # should be lower than MOVED_TOWARD_SAFETY, but at least as high as MOVED_TOWARD_COIN (in magnitude)
    e.COIN_COLLECTED: 100,              # 100 * game reward
    # hunt people
    e.KILLED_OPPONENT: 500,             # 100 * game reward
    MOVED_TOWARD_PLAYER: 1,
    DID_NOT_MOVE_TOWARD_PLAYER: -3,
    # blow up crates
    MOVED_TOWARD_CRATE: 3,
    DID_NOT_MOVE_TOWARD_CRATE: -9,
    # basic stuff
    e.GOT_KILLED: -500,                 # as bad as giving someone else a kill reward
    e.KILLED_SELF: 0,                   # not worse than being killed, so don't punish it (?)
    e.SURVIVED_ROUND: 0,                # dying is already punished, and standing in a corner until the timer runs out should not be rewarded
    e.INVALID_ACTION: -10,
    MOVED_TOWARD_SAFETY: 5,
    DID_NOT_MOVE_TOWARD_SAFETY: -15,
    # be active!
    USELESS_WAIT: -1,                   # it may be good to wait until an explosion is over, so this shouldn't be penalized too much; must not be higher than INVALID_ACTION
    # meaningful bombs
    PLACED_USEFUL_BOMB: 20,
    PLACED_SUPER_USEFUL_BOMB: 50,
    DID_NOT_PLACE_USEFUL_BOMB: -20,     # should be way more than what is gained by running away from the bomb
    e.CRATE_DESTROYED: 0,               # maybe it's bad to reward this because the action that led to this event lies in the past and we're already rewarding good bomb placement
    e.COIN_FOUND: 0,                    # agent cannot influence this, so don't reward it (?)
}

# Some values needed for the potential phi that are not recorded in the game state
bomb_location = None    # location of agent's bomb, if placed
n_crates_total = None   # number of crates at the start of the round

BATCH_SIZE = 128  # number of transitions sampled from replay buffer
MEMORY_SIZE = 5000  # number of transitions to keep in the replay buffer. 5000 is enough for around 35 rounds
GAMMA = 0.99  # discount factor (for rewards in future states)
EPS_START = 0.1  # starting value of epsilon (for taking random actions)
EPS_END = 0.05  # ending value of epsilon
EPS_DECAY = 10  # how many steps until full epsilon decay (not quite true; 'EPS_END' is only attained at infinity)
TAU = 1e-3  # update rate of the target network
LR = 1e-4  # learning rate of the optimizer
OPTIMIZER = optim.Adam  # the optimizer
LAYER_SIZES = [100, 1000, 200, 50]  # sizes of hidden layers

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

# paths to the DQN models
POLICY_MODEL_PATH = f"{cwd}/policy-model.pt"
TARGET_MODEL_PATH = f"{cwd}/target-model.pt"

FEATURE_VECTOR_SIZE = 21  # how many features our model has; ugly but hard to not hardcode

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class Game(TypedDict):
    """For typehints - this is the dictionary we're given by our environment overlords."""
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
    """For storing a defined number of [state + action -> new state + reward] transitions."""

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        # self.rewards = deque([], maxlen=capacity) # more impactful events should be sampled more often

    def push(self, *args):
        self.memory.append(Transition(*args))
        # self.rewards.append(args[-1].item()) # store reward

    def sample(self, batch_size):
        # TODO: better implementation of memory weighing
        # weights = np.array(self.rewards, dtype=float)
        # avg = weights.sum() / weights.shape[0]
        # weights = np.log(np.abs(weights - avg) + 1)
        # # print(f"Weights: {weights}")
        # return random.choices(self.memory, k=batch_size, weights=weights)

        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    """The DQN PyTorch implmenetation."""

    def __init__(self, n_observations: int, n_actions: int, layer_sizes: list[int]):
        super(DQN, self).__init__()

        self.layers = nn.ModuleList(
            [nn.Linear(n_observations, layer_sizes[0])] \
            + [nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)] \
            + [nn.Linear(layer_sizes[-1], n_actions)]
        )

    def forward(self, x: torch.Tensor):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


def _optimize_model(self):
    """
    Performs one optimization of the model by sampling the replay memory.
    See https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html for what this is doing.
    """

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
    torch.nn.utils.clip_grad_value_(self.policy_model.parameters(), 100)  # TODO: to a variable?
    self.optimizer.step()


def _tile_is_free(game_state: Game, x: int, y: int) -> bool:
    """Returns True if a tile is free (i.e. can be stepped on by the player).
    This also returns false if the tile has an ongoing explosion, since while it is free, we can't step there."""
    for obstacle in [p for (p, _) in game_state['bombs']] + [p for (_, _, _, p) in game_state['others']]:
        if obstacle == (x, y):
            return False

    return game_state['field'][x][y] == 0 and game_state['explosion_map'][x][y] == 0


@cache
def _get_blast_coords(x: int, y: int) -> tuple[tuple[int, int]]:
    """For a given bomb at (x, y), return all coordinates affected by its blast."""
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
    If the action is invalid or the player dies, returns None."""
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
        if bomb:
            game_state['bombs'].append(((x, y), s.BOMB_TIMER))
            bomb = False
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

    for i in reversed(range(len(game_state['bombs']))):
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

    # 5. self.evaluate_explosions() - kill agents
    # kill self
    x, y = game_state['self'][3]
    if game_state["explosion_map"][x][y] != 0:
        return None  # we died

    # kill others
    if len(game_state['others']) != 0:
        game_state['others'] = list(game_state['others'])
        for i in reversed(range(len(game_state['others']))):
            x, y = game_state['others'][i][-1]
            if game_state["explosion_map"][x][y] != 0:
                game_state['others'].pop(i)

    return game_state


def _can_escape_after_placement(game_state: Game) -> bool:
    """Return True if the player can escape the bomb blast if it were to place a bomb right now."""
    game_state = copy.copy(game_state)

    x, y = game_state['self'][3]
    game_state['bombs'] = list(game_state['bombs']) + [((x, y), s.BOMB_TIMER)]

    # if it can escape, it's safe
    dist_to_safety = _distances(game_state, search_for={COIN: False, CRATE: False, ENEMY: False, SAFETY: True})[SAFETY]
    return any(dist_to_safety[:4]) # TODO: test if this code does what it should
    # return len(_directions_to_safety(game_state)) != 0


def _is_optimal_position(game_state: Game) -> dict[str, bool]:
    """Evaluates if the agent position is optimal in the given game state in regard to coins, crates, enemies, and safety"""
    pos = game_state['self'][-1]
    out = {COIN: pos in game_state['coins'],
           CRATE: False,
           ENEMY: False,
           SAFETY: not _is_in_danger(game_state)}

    for dx, dy in DELTAS:
        if game_state['field'][pos[0] + dx, pos[1] + dy] == 1:
            out[CRATE] = True
            break

    blast_coords = _get_blast_coords(*pos)
    for enemy in game_state['others']:
        if enemy[-1] in blast_coords:
            out[ENEMY] = True
            break
    return out


def _distances(game_state: Game, search_for: dict[str, bool] | None = None) -> dict[str, list[int]]:
    """Return a dictionary of lists with distances plus 'optimal' signal to the closest coin, enemy and crate, and safety in each direction."""

    # which things may be searched for (coin, crate, enemy, safety)
    if search_for is None:
        search_for = {COIN: len(game_state['coins']) > 0,
                    CRATE: 1 in game_state['field'],
                    ENEMY: len(game_state['others']) > 0,
                    SAFETY: _is_in_danger(game_state)}
    else:
        # make sure the supplied 'search_for' dictionary has the four required entries
        assert all(name in search_for for name in (COIN, CRATE, ENEMY, SAFETY)), 'Supplied "search_for" dictionary does not have the four required entries!'

    # distance in each direction plus 'optimal' signal for each thing
    distances_full:dict[str, list[int]] = {name: [0]*5 for name in search_for}

    agent_pos = game_state['self'][-1]

    # for all four things, check if the agent is already in the optimal position
    for name, is_optimal in _is_optimal_position(game_state).items():
        if is_optimal and search_for[name]:
            distances_full[name][4] = 1 # set 'optimal' signal
    
    # for each neighbor, find closest things via breadth-first search
    for starting_action in ACTIONS[:4]:
        start_game_state = _next_game_state(game_state, starting_action)

        if start_game_state is None: # direction is obstructed or leads to immediate death
            continue

        queue = deque([(start_game_state, 1)])
        explored = {start_game_state['self'][-1], agent_pos} # not allowed to go through agent's position
        distances = {name: 0 for name, sf in search_for.items() if sf} # distance to closest coin/crate/enemy/safety
        # 'distances' also doubles as the break condition for the while loop. That's why it only has entries that are actually being searched for

        while len(queue) > 0:
            current_game_state, distance = queue.popleft()

            if all(distances.values()):
                # found closer things of each type -> BFS is over
                break

            for name, is_optimal in _is_optimal_position(current_game_state).items():
                # if any of the four things are found at the current position, mark their distance and stop searching for it
                if is_optimal and np.isclose(distances.get(name, 1), 0): # 'np.isclose(distances.get(name, 1), 0)' means the thing is being searched for and has not yet been found
                    distances[name] = distance

            # explored neighboring positions
            for action in ACTIONS[:4]:
                next_game_state = _next_game_state(current_game_state, action)

                if next_game_state is not None and next_game_state['self'][-1] not in explored:
                    queue.append((next_game_state, distance + 1))
                    explored.add(next_game_state['self'][-1])
        
        # write 'distances' into 'distances_full'
        i = ACTIONS.index(starting_action)
        for name, distance in distances.items():
            distances_full[name][i] = distance

    # Standing next to a crate counts as being distance 1 from it, and since we're only detecting whether the agent is next to a crate,
    # all valid crate distances need to be increased by 1.
    if search_for[CRATE]:
        for i, (dx, dy) in enumerate(DELTAS):
            if game_state['field'][agent_pos[0] + dx, agent_pos[1] + dy] != -1:
                distances_full[CRATE][i] += 1

    return distances_full


def _is_in_danger(game_state) -> bool:
    """Return True if the player will be killed if it doesn't move (i.e. is in danger)."""
    x, y = game_state['self'][3]
    for ((bx, by), _) in game_state['bombs']:
        if (x, y) in _get_blast_coords(bx, by):
            return True
    return False


def _binarize_features(features:list[int]) -> list[int]:
    """In-place binarization of feature vector"""
    key = lambda x: x if x != 0 else 1e4
    for i in range(4):
        x = 5*i
        y = 5*i+4
        d_min = min(features[x:y], key=key)
        d_max = max(features[x:y], key=key)
        # if all directions are equally good, set them all to 0, otherwise set the best ones to 1
        features[x:y] = [0]*4 if d_min == d_max else [int(d == d_min) for d in features[x:y]]


@lru_cache(maxsize=10000)
def _state_to_features(game_state: tuple) -> torch.Tensor:
    """
    0..4 - distances to closest coins -- u, r, d, l, on top of coin\n
    5..9 - distances to closest crate -- u, r, d, l, next to crate\n
    10..14 - distances to where placing a bomb will hurt another player -- u, r, d, l, place now\n
    15..19 - distances to safety -- u, r, d, l, currently safe\n
    20 - can we place a bomb (and live to tell the tale)?\n

    The features related to coins, crates, enemies, and safety all work exactly the same:
    The first four entries give the distance to the respective thing in the four directions, where immediately going back is not allowed.
    If a distance is obstructed, it's given as 0. For this reason, standing next to a crate gives a distance of 1 (to differentiate between walls and crates).
    The fifth entry states whether the agent is in the optimal spot, i.e. standing on top of a coin (can only happen at the start), standing next to a crate,
    standing in the blast radius of another player or being safe. Note that distances are given even if the agent is in the optimal spot.
    Distances are only given if the respective thing is being searched for. E.g. if the agent is not in danger, no safety distances will be given.
    """
    game_state: Game = {
        'field': np.array(game_state[0]),
        'bombs': list(game_state[1]),
        'explosion_map': np.array(game_state[2]),
        'coins': list(game_state[3]),
        'self': game_state[4],
        'others': list(game_state[5]),
    }

    distances = _distances(game_state)
    can_place_bomb = game_state["self"][2] and _can_escape_after_placement(game_state)

    feature_vector = (distances[COIN]           # coins
                    + distances[CRATE]          # crates
                    + distances[ENEMY]          # enemies
                    + distances[SAFETY]         # safety
                    + [int(can_place_bomb)])    # bomb bit

    # binarize direction features for testing purposes (see line 43)
    if binarize:
        _binarize_features(feature_vector)

    return torch.tensor([feature_vector], device=device, dtype=torch.float)


def state_to_features(game_state: Game | None) -> torch.Tensor | None:
    """A wrapper function so we can cache game states (since you can't cache a dictionary)."""

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


def _is_bomb_useful(game_state) -> bool:
    """Return True if the bomb is useful, either by destroying a crate or by threatening an enemy."""
    x, y = game_state['self'][3]
    for bx, by in _get_blast_coords(x, y):
        # destroys crate
        if game_state['field'][bx][by] == 1:
            return True

        # might kill a player
        if (bx, by) in [a[-1] for a in game_state['others']]:
            return True

    return False


def _is_bomb_location_useful(game_state: Game, loc: tuple[int]) -> bool:
    for bx, by in _get_blast_coords(*loc):
        if game_state['field'][bx, by] == 1:
            return True
    
    return False


def phi(game_state: Game, bomb_location: tuple[int], n_crates) -> float:
    """Potential for game reward: F(s, a, s') = GAMMA*phi(s') - phi(s). See https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf"""
    value = 0.0
    distances_full = _distances(game_state)

    # evaluate distances to things
    key = lambda x: x if x != 0 else 1e4
    importance = {COIN: 1, CRATE: 0.8, ENEMY: 0, SAFETY: 1} # how distances to things should be weighed (arbitrarily chosen for now)
    for name, distances in distances_full.items():
        value += importance[name] * (-min(distances[:4], key=key)) # shorter distance is "less bad"
    
    # evaluate bomb placement TODO: will not work well when enemies are present
    if bomb_location is not None:
        if _is_bomb_location_useful(game_state, bomb_location):
            value += 10 # bomb will destroy crate
        else:
            value -= 5 # bomb will not destroy crate

    # # destroying crates is regarded as a subgoal. It takes roughly t = (COLS-2)*(ROWS-2)/9 + 4 steps to blow up a crate when starting from anywhere
    # # t = (s.COLS - 2)*(s.ROWS - 2)/9 + 4
    # n_crates_remaining = np.sum(game_state['field'] == 1)
    # # value += (n_crates - n_crates_remaining - 0.5) * t / n_crates
    # value -= n_crates_remaining # fewer remaining crates are less bad

    return value


def _reward_from_events(self, events: list[str]) -> torch.Tensor:
    """Utility function for calculating the sum of rewards for events."""
    reward_sum = 0
    for event in events:
        if event in GAME_REWARDS:
            reward_sum += GAME_REWARDS[event]

    self.logger.debug(f"Awarded {reward_sum} for events {', '.join(events)}")

    if self.manual:
        print(f"Awarded {reward_sum} for events {', '.join(events)}")

    return torch.tensor([reward_sum], device=device, dtype=torch.float)


def state_to_string(game_state: Game) -> np.ndarray:
    """Return a string representation of the game state. Useful for debugging"""
    field = game_state['field'].copy()
    field[field == -1] = 10                         # walls are 10, crates are 1
    for c in game_state['coins']:
        field[c] = 3                                # coins are 3
    for b in game_state['bombs']:
        field[b[0]] = -(b[1]+2)                     # bomb detonation time is -4 - -2
    field[game_state['self'][-1]] = 2               # player is 2
    field[game_state['explosion_map'] != 0] = -1    # explosions are -1

    return str(field.T)


def _process_game_event(self, old_game_state: Game, self_action: str,
                        new_game_state: Game | None, events: list[str]):
    """Called after each step when training. Does the training."""

    state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)
    action = torch.tensor([[ACTIONS.index(self_action)]], device=device, dtype=torch.long)

    state_list = state.tolist()[0]

    if not binarize:
        # the features are not yet binarized, so we need to do that to evaluate them
        _binarize_features(state_list)

    moving_events = [
        (MOVED_TOWARD_COIN, DID_NOT_MOVE_TOWARD_COIN, 0, 4),
        (MOVED_TOWARD_CRATE, DID_NOT_MOVE_TOWARD_CRATE, 5, 9),
        (MOVED_TOWARD_PLAYER, DID_NOT_MOVE_TOWARD_PLAYER, 10, 14),
        (MOVED_TOWARD_SAFETY, DID_NOT_MOVE_TOWARD_SAFETY, 15, 19),
    ]

    # generate positive/negative events if we move after the objectives
    for pos_event, neg_event, i, j in moving_events:
        if np.isclose(sum(state_list[i:j]), 0):
            continue

        for i in range(i, j):
            if np.isclose(state_list[i], 1) and self_action == ACTIONS[i % 5]:
                events.append(pos_event)
                break
        else:
            events.append(neg_event)

    # generate positive/negative bomb events if we place a good/bad bomb
    if self_action == "BOMB" and old_game_state['self'][2]:
        if _is_bomb_useful(old_game_state) and state_list[20] == 1:
            # if it endangers a player, it's super useful; otherwise it's just useful
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
            if state_list[20] == 1 and (state_list[9] == 1 or state_list[14] == 1):
                events.append(USELESS_WAIT)

    reward = _reward_from_events(self, events)

    self.memory.push(state, action, new_state, reward)
    self.plot_reward_accum += reward.item()

    _optimize_model(self)

    # soft-update the target network
    target_net_state_dict = self.target_model.state_dict()
    policy_net_state_dict = self.policy_model.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
    self.target_model.load_state_dict(target_net_state_dict)


def _process_game_event_potential(self, old_game_state: Game, self_action: str,
                        new_game_state: Game | None, events: list[str]):
    """Called after each step when training. Does the training. Uses potential-based auxiliary rewards."""
    # auxiliary rewards
    if new_game_state is None: # end of round -> no auxiliary rewards
        reward = 0
    else:
        global bomb_location
        old_bomb_location = bomb_location
        if new_game_state['self'][2]: # if the agent can place a bomb, no bomb is currently ticking
            bomb_location = None
        if e.BOMB_DROPPED:
            bomb_location = new_game_state['self'][-1]

        global n_crates_total
        if n_crates_total is None:
            n_crates_total = np.sum(old_game_state['field'] == 1)

        reward = GAMMA*phi(new_game_state, bomb_location, n_crates_total) - phi(old_game_state, old_bomb_location, n_crates_total)

    # game rewards
    for ev in events:
        if ev == e.COIN_COLLECTED:
            reward += 100
        elif ev == e.KILLED_OPPONENT:
            reward += 500
        # the next ones aren't game rewards, but it's hard to punish them another way (and they should be punished)
        # elif ev == e.GOT_KILLED:
        #     reward -= 500
        # elif ev == e.INVALID_ACTION:
        #     reward -= 15

    state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)
    action = torch.tensor([[ACTIONS.index(self_action)]], device=device, dtype=torch.long)

    self.memory.push(state, action, new_state, torch.tensor([reward], device=device, dtype=torch.float))
    self.plot_reward_accum += reward

    _optimize_model(self)

    # soft-update the target network
    target_net_state_dict = self.target_model.state_dict()
    policy_net_state_dict = self.policy_model.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
    self.target_model.load_state_dict(target_net_state_dict)


def setup_training(self):
    """Sets up training - loads models if they exist + configures plotting (so we see how the model is doing)."""
    if os.path.exists(POLICY_MODEL_PATH):
        self.policy_model = torch.load(POLICY_MODEL_PATH)
    else:
        self.policy_model = DQN(FEATURE_VECTOR_SIZE, len(ACTIONS), LAYER_SIZES).to(device)

    if os.path.exists(TARGET_MODEL_PATH):
        self.target_model = torch.load(TARGET_MODEL_PATH)
    else:
        self.target_model = DQN(FEATURE_VECTOR_SIZE, len(ACTIONS), LAYER_SIZES).to(device)
        self.target_model.load_state_dict(self.policy_model.state_dict())

    self.model = self.policy_model

    self.optimizer = OPTIMIZER(self.policy_model.parameters(), lr=LR)
    self.memory = ReplayMemory(MEMORY_SIZE)

    self.x = []
    self.y = []
    self.plot_reward = []
    self.plot_reward_accum = 0
    self.plot_steps = []

    self.fig = plt.figure(figsize=(6, 3))
    ax = plt.axes()
    self.line1, = ax.plot([], [], color='blue', label='game score')
    self.line2, = ax.plot([], [], color='red', label='reward/100', linestyle='dashed', linewidth=1)
    self.line3, = ax.plot([], [], color='green', label='steps/10')
    ax.legend(loc='lower left')

    plt.show(block=False)


def game_events_occurred(self, old_game_state: Game, self_action: str, new_game_state: Game, events: list[str]):
    """Called once per step to allow intermediate rewards based on game events."""
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if use_potential:
        _process_game_event_potential(self, old_game_state, self_action, new_game_state, events)
    else:
        _process_game_event(self, old_game_state, self_action, new_game_state, events)


def end_of_round(self, last_game_state: dict, last_action: str, events: list[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    if use_potential:
        _process_game_event_potential(self, last_game_state, last_action, None, events)

        # reset values for potential-based rewards
        global bomb_location
        bomb_location = None
        global n_crates_total
        n_crates_total = None
    else:
        _process_game_event(self, last_game_state, last_action, None, events)

    torch.save(self.policy_model, POLICY_MODEL_PATH)
    torch.save(self.target_model, TARGET_MODEL_PATH)

    if len(self.x) > 100:
        self.x.pop(0)
        self.y.pop(0)
        self.plot_reward.pop(0)
        self.plot_steps.pop(0)

    self.x.append(last_game_state['round'])
    self.y.append(last_game_state['self'][1])
    self.plot_reward.append(self.plot_reward_accum / 100)
    self.plot_reward_accum = 0
    self.plot_steps.append(last_game_state['step'] / 10)

    self.line1.set_data(self.x, self.y)
    self.line2.set_data(self.x, self.plot_reward)
    self.line3.set_data(self.x, self.plot_steps)
    self.fig.gca().relim()
    self.fig.gca().autoscale_view()
    self.fig.canvas.draw()
    self.fig.canvas.flush_events()
