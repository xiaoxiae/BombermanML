import argparse
from pathlib import Path
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("task", help="The task to train.")
    parser.add_argument("training_agent", help="The agent to train.")

    arguments = parser.parse_args()

    tasks = {
    "1": [
        f"--no-gui --agents {arguments.training_agent} --scenario coin-heaven --n-rounds 5000",#1000
        f"--no-gui --agents {arguments.training_agent} --scenario sparse-crate --n-rounds 30000",#2000
        f"--no-gui --agents {arguments.training_agent} peaceful_agent --scenario sparse-crate --n-rounds 40000",#2000
        f"--no-gui --agents {arguments.training_agent} peaceful_agent --scenario classic --n-rounds 40000",#2000
        f"--no-gui --agents {arguments.training_agent} peaceful_agent rule_based_agent --scenario classic --n-rounds 40000",#2000
        f"--no-gui --agents {arguments.training_agent} rule_based_agent rule_based_agent rule_based_agent --scenario classic --n-rounds 100000",#2000

    ]
}

    for file in Path(f"agent_code/{arguments.training_agent}").iterdir():
        if file.suffix == ".pk":
            file.unlink()

    for command in tasks[arguments.task]:
        os.system(f"python main.py play --train 1 {command}")
