from collections import namedtuple, deque
from functools import lru_cache, cache

import os
import copy
from itertools import product, permutations
import pickle
#from typing import List
from typing import TypedDict
#from random import shuffle
import random
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt


import events as e
import settings as s


cwd = os.path.abspath(os.path.dirname(__file__))

# path to the QTable models
MODEL_PATH = f"{cwd}/model.pt"

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

FEATURE_VECTOR_SIZE = 12  # how many features our model has; ugly but hard to not hardcode

#MANUAL = False

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

#game rewards to each action of the agent
GAME_REWARDS = {
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
    PLACED_SUPER_USEFUL_BOMB: 150,
    DID_NOT_PLACE_USEFUL_BOMB: -1000,
    e.CRATE_DESTROYED: 10,
    e.COIN_FOUND: 10,
}

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

# Actions
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
STATE_SPACE = (17,17)
DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

#Training parameters
N_TRAINING_EPSIDOES = 10000
LEARNING_RATE = 0.7
#Evaluation_parameters
N_EVAL_EPISODES = 100
#Environment parameters
ENV_ID = "Bombermann"
MAX_STEPS = 400
GAMMA = 0.95
EVAL_SEED = []
N_EPISODES = 10
#Exploration parameters
MAX_EPSILON = 1.0
MIN_EPSILON = 0.05
DECAY_RATE = 0.0005



class Game(TypedDict):
    """For typehints - this is the dictionary we're given by our environment overlords."""
    field: np.ndarray
    bombs: list[tuple[tuple[int, int], int]]
    explosion_map: np.ndarray
    coins: list[tuple[int, int]]
    self: tuple[str, int, bool, tuple[int, int]]
    others: list[tuple[str, int, bool, tuple[int, int]]]
    step: int
    round: int
    # these aren't really important, so we don't expect them to be there
    # user_input: str | None

class QTable():
    """
        the structure of the q_table is a three linked dictionaries to reduce the domain of search
        {state:{substate:{actions: probability}}} --> e.g: {(1,1):{feature:{'LEFT':0.95}}},
        by this approach we can reduce the domain of search to substates that are just neighbour to our agent,
        max 4* 2 ^ features
        Features are binary
        one example of one row from q_table:
        {(0,1): {(0, 0, 0, 0, 0, 0, 0, 0, 0, 0): {'UP': 0,'RIGHT': 0,'DOWN': 0,'LEFT': 0,'WAIT': 0,'BOMB': 0}
        
    """
    def __init__(self, game_state: Game):
        super(QTable, self).__init__()
        self.game_state = game_state
    
    def initialize_q_table(self) -> dict:
        """initializing an empty qtable
        Returns:
            np.ndarray: _description_
        """
        #state_space = self.game_state['field'].size
        # action_space = len(ACTIONS)
        # model = np.zeros((np.power(2, FEATURE_VECTOR_SIZE) ,action_space))
        
        model = {}
        # creating a linked dictionary as a q_table
        # actions_dict = dict.fromkeys(ACTIONS,0)
        # features_list = list(product([0,1],repeat=10))
        # features_dict = dict.fromkeys(features_list, actions_dict)
        # dimention = [i for i in range(0,17)]
        # perm = permutations(dimention,2)
        # q_table = dict.fromkeys(perm, features_dict)

        return model

def _tile_is_free(game_state: Game, x: int, y: int) -> bool:
    """Returns True if a tile is free (i.e. can be stepped on by the player).
    This also returns false if the tile has an ongoing explosion, since while it is free, we can't step there."""
    for obstacle in [p for (p, _) in game_state['bombs']] + [p for (_, _, _, p) in game_state['others']]:
        if obstacle == (x, y):
            return False

    return game_state['field'][x][y] == 0 and game_state['explosion_map'][x][y] == 0

@cache
def _get_blast_coords( x: int, y: int) -> tuple[tuple[int, int]]:
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

def _reward_from_events(self, events: list[str]) -> list:
    """Utility function for calculating the sum of rewards for events."""
    reward_sum = 0
    for event in events:
        if event in GAME_REWARDS:
            reward_sum += GAME_REWARDS[event]

    self.logger.debug(f"Awarded {reward_sum} for events {', '.join(events)}")

    # if MANUAL:
    #     print(f"Awarded {reward_sum} for events {', '.join(events)}")

    return reward_sum

def _next_game_state(game_state: Game, action: str) -> Game | None:
    """Return a new game state by progressing the current one given the action.
    Assumes that all other players stand perfectly still.
    If the action is invalid or the player dies, returns None."""
    game_state = copy.copy(game_state)
    game_state['bombs'] = list(game_state['bombs'])

    # 1. self.poll_and_run_agents() - only moves
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
    # kill self
    x, y = game_state['self'][3]
    if game_state["explosion_map"][x][y] != 0:
        return None  # we died

    # kill others
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
    """Return True if the player can escape the bomb blast if it were to place a bomb right now."""
    game_state = copy.copy(game_state)

    x, y = game_state['self'][-1]
    game_state['bombs'] = list(game_state['bombs']) + [((x, y), s.BOMB_TIMER)]

    # if it can escape, it's safe
    return len(_directions_to_safety(game_state)) != 0


def _directions_to_coins(game_state: Game) -> list[int]:
    """Return a list with directions to the closest coin."""
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
            # if we're standing on it, return 4 (i.e. wait)
            if current == start:
                return [4]

            # otherwise backtrack
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

def _directions_to_enemy(game_state: Game) ->list[int]:
    """Return a list with directions to the closest enemy."""
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

        for n in current_game_state['others']:
            # if placing a bomb would kill another player, we're here
            if n[-1] in _get_blast_coords(*current):
                # if we're at the start, index 4 signals "place a bomb now"
                if current == start:
                    return [4]

                while explored[current] != start:
                    current = explored[current]

                candidates.add(DELTAS.index((current[0] - start[0], current[1] - start[1])))
                candidate_distance = distance

                break

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

    return list(candidates)


def _directions_to_crates(game_state: Game) -> list[int]:
    """Return a list with directions to the closest crate."""
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

def _is_in_danger(game_state) -> bool:
    """Return True if the player will be killed if it doesn't move (i.e. is in danger)."""
    x, y = game_state['self'][-1]
    for ((bx, by), _) in game_state['bombs']:
        if (x, y) in _get_blast_coords(bx, by):
            return True
    return False

def _directions_to_safety(game_state) -> list[int]:
    """Return the directions to safety, if the player is currently in danger of dying."""

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
def _state_to_features(game_state: tuple | None) -> list | None:
    """
    # 0,1 self position
    # 2..6 - direction to closest coin -- u, r, d, l, wait and direction to safety; has a one only if is in danger -- u, r, d, l, wait
    # 7..11 - direction to closest crate -- u, r, d, l, wait and direction to where placing a bomb will hurt another player -- u, r, d, l, place now 
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
    feature_vector[0] = game_state['self'][3][0]
    feature_vector[1] = game_state['self'][3][1]

    if v := _directions_to_coins(game_state):
        for i in v:
            feature_vector[i + 2] = 1

    if v := _directions_to_crates(game_state):
        for i in v:
            feature_vector[i + 7] = 1

    if v := _directions_to_enemy(game_state):
        for i in v:
            feature_vector[i + 7] = 1

    if v := _directions_to_safety(game_state):
        for i in v:
            feature_vector[i + 2] = 1

        # if we can get to safety by something other than waiting, don't wait
        if v != [6]:
            feature_vector[6] = 0

        # if we need to run away, mask other features to do that too
        if 1 in [feature_vector[p + 2] for p in range(5) ]:
            for j in range(5):
                feature_vector[j + 7] &= 1
    else:
        # TODO: directions away from the bomb
        #  if an enemy is blocking the way, it's still better to just run away from the bomb
        #  the code is probably the same but we ignore all player positions
        pass

    if game_state["self"][2] and _can_escape_after_placement(game_state):
        feature_vector[11] = 1

    # # feature 14 is 'place a bomb to kill player' so that needs to be masked with 20
    # feature_vector[14] &= feature_vector[20]

    return feature_vector


def state_to_features(game_state: Game | None) -> list | None:
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
    """Return True if the bomb is useful, either by destroying a crate or by killing an enemy."""
    x, y = game_state['self'][3]
    for bx, by in _get_blast_coords(x, y):
        # destroys crate
        if game_state['field'][bx][by] == 1:
            return True

        # kills a player
        if (bx, by) in [a[-1] for a in game_state['others']]:
            return True

    return False



def _process_game_event(self, old_game_state: Game, self_action: str,
                        new_game_state: Game | None, events: list[str]):
    """Called after each step when training. Does the training."""
    state = state_to_features(old_game_state)
    # new_state = state_to_features(new_game_state)
    # action = [ACTIONS.index(self_action)]

    #state_list = state.tolist()[0]

    moving_events = [
        (MOVED_TOWARD_COIN, DID_NOT_MOVE_TOWARD_COIN, 2, 6),
        (MOVED_TOWARD_CRATE, DID_NOT_MOVE_TOWARD_CRATE, 7, 11),
        (MOVED_TOWARD_PLAYER, DID_NOT_MOVE_TOWARD_PLAYER, 7, 11),
        (MOVED_TOWARD_SAFETY, DID_NOT_MOVE_TOWARD_SAFETY, 2, 6),
    ]

    # generate positive/negative events if we move after the objectives
    for pos_event, neg_event, i, j in moving_events:
        if np.isclose(sum(state[i:j]), 0):
            continue

        for i in range(i, j):
            if np.isclose(state[i], 1) and self_action == ACTIONS[i % 5]:
                events.append(pos_event)
                break
        else:
            events.append(neg_event)

    # 14 means 'place a bomb to kill player' and not 'wait'
    if state[11] == 1:
        if self_action == 'WAIT' and MOVED_TOWARD_PLAYER in events:
            events.remove(MOVED_TOWARD_PLAYER)
        elif self_action == 'BOMB' and DID_NOT_MOVE_TOWARD_PLAYER in events:
            events.remove(DID_NOT_MOVE_TOWARD_PLAYER)

    # generate positive/negative bomb events if we place a good/bad bomb
    if self_action == "BOMB" and old_game_state['self'][2]:
        if _is_bomb_useful(old_game_state) and state[11] == 1:
            # if it endangers a player, it's super useful; otherwise it's just useful
            if state[11]:
                events.append(PLACED_SUPER_USEFUL_BOMB)
            else:
                events.append(PLACED_USEFUL_BOMB)
        else:
            events.append(DID_NOT_PLACE_USEFUL_BOMB)

    # # if we wait, make sure it's meaningful (i.e. we weren't recommended to move somewhere)
    # if self_action == "WAIT":
    #     # waiting near a crate / player when we can place a bomb is also useless
    #     if state[20] == 1 and (state[9] == 1 or state[14] == 1):
    #         events.append(USELESS_WAIT)
    #     else:
    #         for i in [j + 5 * i for i in range(3) for j in range(4)]:
    #             if state[i] == 1:
    #                 events.append(USELESS_WAIT)
    #                 break

    reward = _reward_from_events(self, events)

    self.total_reward += reward

    # self.memory.push(state, action, new_state, reward)


#udate our model here
    self.model =_update_model(self, old_game_state)


def _epsilon_greedy_policy( model: dict, state: list,  epsilon: float) -> str:
    """
With a Probability of 1 - ɛ, we do exploitation, and with the probability ɛ,
we do exploration. 
In the epsilon_greedy_policy we will:
1-Generate the random number between 0 to 1.
2-If the random number is greater than epsilon, we will do exploitation.
    It means that the agent will take the action with the highest value given
    a state.
3-Else, we will do exploration (Taking random action). 

"""
    random_int = random.uniform(0,1)
    if random_int > epsilon:
        action = _greedy_policy(model,state)
    else:
        action = random.choice(ACTIONS)
        # action = ACTIONS.index(action)
        #action = ACTIONS[action]
    return action

def _greedy_policy(model: dict, state: list) -> str:
    """
Q-learning is an off-policy algorithm which means that the policy of 
   taking action and updating function is different.
In this example, the Epsilon Greedy policy is acting policy, and 
   the Greedy policy is updating policy.
The Greedy policy will also be the final policy when the agent is trained.
   It is used to select the highest state and action value from the Q-Table.
"""
    state = tuple(state)
    action = np.argmax(model[state])
    action = ACTIONS[action]
    return action


def _update_model(self, game_state: Game) ->np.ndarray:
    """Training the agent to update the qtable

    Returns:
        np.ndarray: _description_
    """
    # epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON)*np.exp(-DECAY_RATE*game_state['round'])
    # feature = state_to_features(game_state)
    #state = {game_state['self']:feature}

    for episode in trange(game_state['round']):#n_training_episodes must be taken from game_state
        epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON)*np.exp(-DECAY_RATE * episode)
        #Reset the environment
        #state = game_state.reset()[0]# need a function to reset the game and recieve the new state of agent
        state = state_to_features(game_state)
        state = tuple(state)
        #done = False
        
        #repeat
        for _ in range(MAX_STEPS):
            action = _epsilon_greedy_policy(self.model, state, epsilon)# current state of agent is needed
            #new_state, reward, done, info, _ = game_state.step(action)# features to reward function is needed
            new_state = state_to_features(_next_game_state(game_state, action))
            reward = self.total_reward
            if not self.model.get(state):
                self.model[state] = 0 #  if no Key create the key
            else:

                if self.model[state][action] and new_state:
                    new_state = tuple(new_state)
                    self.model[state][action] = self.model[state][action] + LEARNING_RATE*(
                        reward + GAMMA * np.max(self.model[new_state]) - self.model[state][action])#current and previous state of agent 
                else:
                    self.model[state][action] = LEARNING_RATE * reward
                #if done, finish the episode
                #if done:
                if game_state:
                    break
                    
                #update state
                state = new_state
    return self.model


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.total_reward = 0

    self.model = QTable.initialize_q_table(self)

    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as file:
            self.model = pickle.load(file)

    self.x = [0]
    self.y_score = [0]
    self.y_reward = [0]
    self.y_steps = [0]

    self.fig = plt.figure(figsize=(6, 3))
    ax = plt.axes()

    self.plot_score, = ax.plot(self.x, self.y_score, '-', color='blue', label='game score')
    self.plot_reward, = ax.plot(self.x, self.y_reward, color='red', label='reward/100', linestyle='dashed', linewidth=1)
    self.plot_steps, = ax.plot(self.x, self.y_steps, '-', color='green', label='steps/40')
    ax.legend(loc='lower left')

    plt.show(block=False)


def game_events_occurred(self, old_game_state: Game, self_action: str, new_game_state: Game, events: list[str]):
    """Called once per step to allow intermediate rewards based on game events."""
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    _process_game_event(self, old_game_state, self_action, new_game_state, events)


def end_of_round(self, last_game_state: Game, last_action: str, events: list[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    _process_game_event(self, last_game_state, last_action, None, events)

    if len(self.x) > 100:
        self.x.pop(0)
        self.y_score.pop(0)
        self.y_reward.pop(0)
        self.y_steps.pop(0)

    self.x.append(self.x[-1] + 1)
    self.y_score.append(last_game_state['self'][1])
    self.y_reward.append(self.total_reward / 1000)
    self.y_steps.append(last_game_state['step'] / 40)

    self.plot_score.set_data(self.x, self.y_score)
    self.plot_reward.set_data(self.x, self.y_reward)
    self.plot_steps.set_data(self.x, self.y_steps)

    self.total_reward = 0

    self.fig.gca().relim()
    self.fig.gca().autoscale_view()
    self.fig.canvas.draw()
    self.fig.canvas.flush_events()

    # Store the model
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(self.model, file)

