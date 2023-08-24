import math
from random import choice

from .train import *

cwd = os.path.abspath(os.path.dirname(__file__))


def setup(self):
    if not self.train:
        self.model = torch.load(f"{cwd}/policy-model.pt")


def act(self, game_state: Game) -> str:
    """Perform the most probable action when in practice, otherwise use epsilon-decay for better training."""
    if self.train:
        steps = game_state["step"]

        threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps / EPS_DECAY)

        if random.random() <= threshold:
            action = choice(ACTIONS)
            self.logger.debug(f"Picking random action of {action}.")
            return action

    with torch.no_grad():
        model_result = self.model(state_to_features(game_state))

    action = ACTIONS[model_result.max(1)[1].item()]
    self.logger.debug(f"Action weights: {model_result}, picking {action}.")
    return action
