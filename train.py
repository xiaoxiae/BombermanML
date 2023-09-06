import argparse
import shutil
from pathlib import Path
import os

TASKS = {
    "1": [
        "--no-gui --agents user_agent --scenario coin-heaven --n-rounds 100",
    ],
    "complete": [
        "--no-gui --agents user_agent --scenario coin-heaven --n-rounds 100",
        "--no-gui --agents user_agent rule_based_agent --scenario empty --n-rounds 1000",
        "--no-gui --agents user_agent --scenario classic --n-rounds 1000",
        "--no-gui --agents user_agent rule_based_agent --scenario classic --n-rounds 10000",
    ]
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("agent", help="The agent to train.")
    parser.add_argument("--task", help="The task to train. Defaults to 'complete', which trains for all tasks.",
                        default='complete')
    parser.add_argument("--continue", help="Continue training (instead of removing the network).", action="store_true")

    arguments = parser.parse_args()

    agent_directory = Path(f"agent_code/{arguments.agent}")
    agent_model_directory = agent_directory / "model"

    # remove the current network (a new one will be trained)
    if not vars(arguments)['continue']:
        for file in agent_directory.iterdir():
            if file.suffix == ".pt":
                file.unlink()

    # run the training commands
    for i, command in enumerate(TASKS[arguments.task]):
        print(f"Running '{command}'")
        os.system(f"python main.py play --train 1 {command}")

        print(f"Saving current network state")
        agent_model_directory.mkdir(exist_ok=True)
        for file in agent_directory.iterdir():
            if file.suffix == ".pt":
                shutil.copy(str(file), (agent_model_directory / file.name).with_suffix(f"_{i}.pt"))
