import math
from random import choice

from .train import *

cwd = os.path.abspath(os.path.dirname(__file__))


def setup(self):
    self.manual = False

    if not self.train and not self.manual:
        self.model = DQN(FEATURE_VECTOR_SIZE, len(ACTIONS), LAYER_SIZES).to(device)
        self.model.load_state_dict(torch.load(TARGET_MODEL_PATH))
        self.model.eval()


def act(self, game_state: Game) -> str:
    """Perform the most probable action when in practice, otherwise use epsilon-decay for better training."""
    if self.manual:
        state = state_to_features(game_state)
        print(r"  /       coin        \    /       crate       \    /      player       \    /       safety      \ ")
        print(r" /u    r    d    l    w\  /u    r    d    l    w\  /u    r    d    l    b\  /u    r    d    l    w\  /b\ ")
        print(state.tolist()[0])
        return game_state['user_input']

    if self.train:
        self.logger.debug("\n=========")
        round = game_state['round']

        threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * round / EPS_DECAY)
        if game_state['step'] == 1: self.logger.debug(f"Randomness level at {threshold*100:.2f}%")

        if random.random() <= threshold:
            action = choice(ACTIONS)
            self.logger.info(f"Picking random action of {action}.")
            return action

    with torch.no_grad():
        state = state_to_features(game_state)
        model_result = self.model(state)

    action_weights = model_result.softmax(1)[0].tolist()
    action = random.choices(ACTIONS, action_weights)[0] # softmax
    # action = ACTIONS[model_result.max(1)[1].item()]   # greedy
    self.logger.info(f"Picking {action} from state {state}.")
    return action
