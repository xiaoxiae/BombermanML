from collections import  deque
from functools import lru_cache, cache

import os
import copy
from itertools import product
import pickle
from typing import TypedDict
import random
import numpy as np
import matplotlib.pyplot as plt


import events as e
import settings as s


cwd = os.path.abspath(os.path.dirname(__file__))

# path to the QTable models
MODEL_PATH = f"{cwd}/model.pkl"

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

FEATURE_VECTOR_SIZE = 11  # Number of features
ZERO = 0.0

# Field to reduce the number of features
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
    MOVED_TOWARD_COIN: 2,
    DID_NOT_MOVE_TOWARD_COIN: -6,
    e.COIN_COLLECTED: 100,
    # hunt people
    e.KILLED_OPPONENT: 500,
    MOVED_TOWARD_PLAYER: 1,
    DID_NOT_MOVE_TOWARD_PLAYER: -3,
    # blow up crates
    MOVED_TOWARD_CRATE: 3,
    DID_NOT_MOVE_TOWARD_CRATE: -9,
    # basic stuff
    e.GOT_KILLED: -500,
    e.KILLED_SELF: 0,
    e.SURVIVED_ROUND: 0,
    e.INVALID_ACTION: -10,
    MOVED_TOWARD_SAFETY: 5,
    DID_NOT_MOVE_TOWARD_SAFETY: -15,
    # be active!
    USELESS_WAIT: -1,
    # meaningful bombs
    PLACED_USEFUL_BOMB: 20,
    PLACED_SUPER_USEFUL_BOMB: 50,
    DID_NOT_PLACE_USEFUL_BOMB: -20,
    e.CRATE_DESTROYED: 0,
    e.COIN_FOUND: 0,
}


# Actions
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

#Training parameters
LEARNING_RATE = 0.7
#Environment parameters
MAX_STEPS = 400
GAMMA = 0.95
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

class QTable():
    """
        the structure of the q_table is a linked dictionary to reduce the domain of search
        {state:{actions: Q_value}} --> e.g: {feature:{'LEFT':-0.95}},
        Features are binary
        one example of one row from q_table at time of initializing:
        {(0,1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0): {'UP': 0,'RIGHT': 0,'DOWN': 0,'LEFT': 0,'WAIT': 0,'BOMB': 0}
        
    """
    def __init__(self, game_state: Game):
        super(QTable, self).__init__()
        self.game_state = game_state
    
    def initialize_q_table(self) -> dict:
        """initializing an empty qtable
        Returns:
            dict: dictionary of {state:{actions: Q_value}}
        """
        features_list = list(product([0, 1], repeat=10))
        dim = []
        for i in range(len(EMPTY_FIELD)):
            for j in range(len(EMPTY_FIELD)):
                if EMPTY_FIELD[i,j] == 0:
                    dim.append((i,j))

        features = []
        for p in dim:
            for item in features_list:
                features.append(p + item)

        features_dict = {}
        for p in dim:
            feature_keys = [((p,) + feature) for feature in features_list]  # Add p as a single element to the key
            for key in feature_keys:
                features_dict[key] = dict.fromkeys(ACTIONS, 0.0)  # Initialize each nested dictionary

        return features_dict

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

def _reward_from_events(self, events: list[str]) -> list:
    """Utility function for calculating the sum of rewards for events."""
    reward_sum = 0
    for event in events:
        if event in GAME_REWARDS:
            reward_sum += GAME_REWARDS[event]

    self.logger.debug(f"Awarded {reward_sum} for events {', '.join(events)}")

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
            # if n[-1]  in _get_blast_coords(*current):
            if n[-1] in _get_blast_coords(*current):
                # if we're at the start, index 5 signals "place a bomb now"
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
                # if we're at the start, index 5 signals "place a bomb now"
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
    # actions = [ACTIONS.index(action) for action in valid_actions]
    return [ACTIONS.index(action) for action in valid_actions]

@lru_cache(maxsize=10000)
def _state_to_features(game_state: tuple | None) -> list | None:
    """
    # 0 self position
    # 2..5 - direction to closest coin -- u, r, d, l, wait and direction to safety; has a one only if is in danger -- u, r, d, l, wait
    # 6..10 - direction to closest crate -- u, r, d, l, place a bomb and direction to where placing a bomb will hurt another player -- u, r, d, l, place now 
    
    """
    if game_state is None:
        return None
    game_state: Game = {
        'field': np.array(game_state[0]),
        'bombs': list(game_state[1]),
        'explosion_map': np.array(game_state[2]),
        'coins': list(game_state[3]),
        'self': game_state[4],
        'others': list(game_state[5]),
    }

    feature_vector = [0] * FEATURE_VECTOR_SIZE
    feature_vector[0] = game_state['self'][3]
    

    if v := _directions_to_coins(game_state):
        for i in v:
            feature_vector[i + 1] = 1

    if v := _directions_to_crates(game_state):
        for i in v:
            feature_vector[i + 6] = 1

    if v := _directions_to_enemy(game_state):
        for i in v:
            feature_vector[i + 6] = 1

    if v := _directions_to_safety(game_state):
        for i in v:
            feature_vector[i + 1] = 1

        # # if we can get to safety by something other than waiting, don't wait
        # if v != [5]:
        #     feature_vector[5] = 0

        # if we need to run away, mask other features to do that too
        if 1 in v: # [feature_vector[p + 1] for p in range(5) ]
            for j in range(4):
                feature_vector[j + 6] &= 1
            feature_vector[10] = 0
    else:
        # TODO: directions away from the bomb
        #  if an enemy is blocking the way, it's still better to just run away from the bomb
        #  the code is probably the same but we ignore all player positions
        pass

    if game_state["self"][2] and _can_escape_after_placement(game_state):
        feature_vector[10] = 1

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

    moving_events = [
        (MOVED_TOWARD_COIN, DID_NOT_MOVE_TOWARD_COIN, 1, 5),
        (MOVED_TOWARD_CRATE, DID_NOT_MOVE_TOWARD_CRATE, 6, 10),
        (MOVED_TOWARD_PLAYER, DID_NOT_MOVE_TOWARD_PLAYER, 6, 10),
        (MOVED_TOWARD_SAFETY, DID_NOT_MOVE_TOWARD_SAFETY, 1, 5),
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

    # 10 means 'place a bomb to kill player' and not 'wait'
    if state[10] == 1:
        if self_action == 'WAIT' and MOVED_TOWARD_PLAYER in events:
            events.remove(MOVED_TOWARD_PLAYER)
        elif self_action == 'BOMB' and DID_NOT_MOVE_TOWARD_PLAYER in events:
            events.remove(DID_NOT_MOVE_TOWARD_PLAYER)

    # generate positive/negative bomb events if we place a good/bad bomb
    if self_action == "BOMB" and old_game_state['self'][2]:
        if _is_bomb_useful(old_game_state) and state[10] == 1:
            # if it endangers a player, it's super useful; otherwise it's just useful
            if state[10]:
                events.append(PLACED_SUPER_USEFUL_BOMB)
            else:
                events.append(PLACED_USEFUL_BOMB)
        else:
            events.append(DID_NOT_PLACE_USEFUL_BOMB)

    reward = _reward_from_events(self, events)

    self.total_reward += reward


#udate our model here
    self.model =_update_model(self,old_game_state, state, self_action)


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
    if state and random_int > epsilon:
        action = _greedy_policy(model,state)
    else:
        action = random.choice(ACTIONS)
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


def _update_model(self,game_state: Game, state: list | None, action: str| None) ->np.ndarray:
    """Training the agent to update the qtable

    Returns:
        np.ndarray: _description_
    """
    epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON)*np.exp(-DECAY_RATE * game_state['step'])
        # end of the game
    if game_state is None:
        pass

    if state:# if state is not None
        state = tuple(state)
    if not action : # if action is None go for random action
        action = _epsilon_greedy_policy(self.model, state, epsilon)
    # find the new state according to previous one and the action in QTable
    new_state = state_to_features(_next_game_state(game_state, action))
    reward = self.total_reward

    if (self.model[state][action] is not None) and new_state: # if action is valid, update the Qvalue of that state_action
        
        new_state = tuple(new_state)
        model_new_result = self.model[new_state]
        max_result = max(model_new_result.values())

        self.model[state][action] = self.model[state][action] + LEARNING_RATE*(reward + GAMMA * max_result - self.model[state][action])

    elif new_state is None:
        self.model[state][action] = self.model[state][action] + LEARNING_RATE*(reward + GAMMA * ZERO - self.model[state][action])

    return self.model

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.total_reward = 0

    

    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as file:
            self.model = pickle.load(file)
    else:
        self.model = QTable.initialize_q_table(self)


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

