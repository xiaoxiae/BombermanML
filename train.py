import argparse
from pathlib import Path
import os

TASKS = {
    "1": [
        "--no-gui --agents user_agent --scenario coin-heaven --n-rounds 1000",
    ],
    "2": [
        "--no-gui --agents user_agent --scenario classic --n-rounds 10000",
    ]
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("task", help="The task to train.")

    arguments = parser.parse_args()

    for file in Path("agent_code/user_agent").iterdir():
        if file.suffix == ".pt":
            file.unlink()

    for command in TASKS[arguments.task]:
        os.system(f"python3 main.py play --train 1 {command}")
