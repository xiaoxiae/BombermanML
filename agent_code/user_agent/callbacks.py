import math
from random import choice

from .train import *

cwd = os.path.abspath(os.path.dirname(__file__))


def setup(self):
    self.manual = False

    if not self.train:
        self.model = torch.load(POLICY_MODEL_PATH)


def act(self, game_state: Game) -> str:
    """Perform the most probable action when in practice, otherwise use epsilon-decay for better training."""
    if self.manual:
        state = state_to_features(game_state)
        print(r"  /       coin        \    /       crate       \    /      player       \    /       safety      \ ")
        print(r" /u    r    d    l    w\  /u    r    d    l    w\  /u    r    d    l    w\  /u    r    d    l    w\  /b\ ")
        print(state.tolist()[0])
        return game_state['user_input']

    if self.train:
        steps = game_state["step"]

        threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps / EPS_DECAY)

        if random.random() <= threshold:
            action = choice(ACTIONS[:5] * 5 + ACTIONS[5:])
            self.logger.info(f"Picking random action of {action}.")
            return action

    with torch.no_grad():
        state = state_to_features(game_state)
        model_result = self.model(state)

    action = ACTIONS[model_result.max(1)[1].item()]
    self.logger.info(f"Picking {action} from state {state}.")
    return action
