import argparse
import shutil
import subprocess
from pathlib import Path
import os

# the lists are tuple (training command, calculate the agent elo)
# if None is specified, it's the name of the agent
TASKS = {
    "1": [
        (["--scenario", "coin-heaven", "--n-rounds", "100"], False),
    ],
    "complete": [
        (["--scenario", "coin-heaven", "--n-rounds", "100"], False),
        (["rule_based_agent", "--scenario", "empty", "--n-rounds", "1000"], False),
        (["--scenario", "classic", "--n-rounds", "1000"], False),
        (["rule_based_agent", "--scenario", "classic", "--n-rounds", "200"], True),
        (["binary_agent_v3", "--scenario", "classic", "--n-rounds", "200"], True),
        ([None, "--scenario", "classic", "--n-rounds", "200"], True),
    ],
    "self": [
        ([None, "--scenario", "classic", "--n-rounds", "100"], True),
    ],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("agent", help="The agent to train.")
    parser.add_argument("--task", help="The task to train. Defaults to 'complete', which trains for all tasks.",
                        default='complete')
    parser.add_argument("--continue", help="Continue training (instead of removing the network).", action="store_true")
    parser.add_argument("--infinite", help="Train indefinitely, repeating the last n commands.", type=int)

    arguments = parser.parse_args()

    agent_directory = Path(f"agent_code/{arguments.agent}")
    agent_model_directory = agent_directory / "models"

    # remove the current network (a new one will be trained)
    if not vars(arguments)['continue']:
        for file in agent_directory.iterdir():
            if file.suffix == ".pt":
                file.unlink()

    # yeah, I know, I'm going to hell
    if arguments.infinite:
        TASKS[arguments.task] += TASKS[arguments.task][-arguments.infinite:] * 10000

    # run the training commands
    for i, (command, calculate_elo) in enumerate(TASKS[arguments.task]):
        # Nones are the names of the agents
        command = [c or arguments.agent for c in command]

        result_command = ["python", "main.py", "play", "--train", "1", "--no-gui", "--agents",
                          arguments.agent] + command

        print(f"Running '{' '.join(result_command)}'")
        subprocess.Popen(result_command).communicate()

        print(f"Saving current network state")
        agent_model_directory.mkdir(exist_ok=True)
        for file in agent_directory.iterdir():
            # save .pt (networks) and .txt (other data)
            for suffix in [".pt", ".txt"]:
                if file.suffix == suffix:
                    shutil.copy(str(file), (agent_model_directory / file.name).with_suffix(f"{suffix}.{i}"))

        if calculate_elo:
            print("Calculating the agent's elo against the others.")

            result = subprocess.Popen(
                ["python", "elo.py", "-n", "100", "--no-save", "agent", arguments.agent],
                stdout=subprocess.PIPE,
            ).communicate()

            elo = result[0].decode()
            print(elo)

            with open(agent_model_directory / f"elo.txt.{i}", "w") as f:
                f.write(elo)
