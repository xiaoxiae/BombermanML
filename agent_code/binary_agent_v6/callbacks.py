import math
from random import choice

from .train import *

cwd = os.path.abspath(os.path.dirname(__file__))


def setup(self):
    if not self.train and not MANUAL:
        self.model = DQN(FEATURE_VECTOR_SIZE, len(ACTIONS), LAYER_SIZES).to(device)
        self.model.load_state_dict(torch.load(TARGET_MODEL_PATH, map_location=device))
        self.model.eval()


def act(self, game_state: Game) -> str:
    """Perform the most probable action when in practice, otherwise use epsilon-decay for better training."""
    if MANUAL:
        state = state_to_features(game_state)
        print(r"  /       coin        \    /       crate       \    /      player       \    /       safety      \ ")
        print(r" /u    r    d    l    w\  /u    r    d    l    w\  /u    r    d    l    b\  /u    r    d    l    w\  /b\ ")
        print(state.tolist()[0])
        return game_state['user_input']

    if self.train:
        threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * game_state["step"] / EPS_DECAY)

        if random.random() <= threshold:
            action = choice(ACTIONS)
            self.logger.info(f"Picking random action of {action}.")
            return action

    with torch.no_grad():
        state = state_to_features(game_state)
        model_result = self.model(state)

    action = ACTIONS[model_result.max(1)[1].item()]
    self.logger.info(f"Picking {action} from state {state}.")
    return action
