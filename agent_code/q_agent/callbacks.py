
from .train import *


import pickle
import random

cwd = os.path.abspath(os.path.dirname(__file__))


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if not self.train:
        self.logger.info("Loading model from saved state.")
        with open(MODEL_PATH, "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: Game) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    def _random_action(self) -> str:
        """choosing random action when there is no state in the model or we are in the training mode

        Returns:
            action: _description_
        """
        epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON)*np.exp(-DECAY_RATE*game_state['step'])
        random_int = random.uniform(0,1)
        if random_int <= epsilon:
            action = random.choice(ACTIONS)
            self.logger.debug("Choosing action purely at random.")
            return action
        else:
            action = random.choice(ACTIONS)
    

    state = state_to_features(game_state)
    state = tuple(state)
    # if self.train and not self.model.get(state):
    #     return _random_action(self)
    
    if  self.model.get(state):
        model_result = self.model[state]
    else:
        model_result = None
    if model_result:# if there is any value for the state in q_table
        #action = max(zip(model_result.values(), model_result.keys()))[1]
        #choosing the best action acording to the q_table
        action = np.argmax(model[state])
        action = ACTIONS[action]
        self.logger.info(f"Picking {action} from state {state}.")
        return action
    else:
        if self.train and not self.model.get(state):
            return _random_action(self)

