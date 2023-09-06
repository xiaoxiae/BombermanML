import argparse
from pathlib import Path
import os


# TASKS = {
#     "1": [
#         "--no-gui --agents distance_agent --scenario coin-heaven --n-rounds 100",
#     ],
#     "2": [
#         "--no-gui --agents distance_agent --scenario coin-heaven --n-rounds 50",
#         "--no-gui --agents distance_agent peaceful_agent --scenario empty --n-rounds 1000",
#         "--no-gui --agents distance_agent peaceful_agent --scenario classic --n-rounds 10000",
#     ]
# }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("task", help="The task to train.")
    parser.add_argument("training_agent", help="The agent to train.")

    arguments = parser.parse_args()

    tasks = {
    "1": [
        f"--no-gui --agents {arguments.training_agent} --scenario coin-heaven --n-rounds 100",
    ],
    "2": [
        f"--no-gui --agents {arguments.training_agent} --scenario coin-heaven --n-rounds 100",
        f"--no-gui --agents {arguments.training_agent} peaceful_agent --scenario empty --n-rounds 1000",
        #f"--no-gui --agents {arguments.training_agent} peaceful_agent --scenario classic --n-rounds 10000",
    ]
}

    for file in Path(f"agent_code/{arguments.training_agent}").iterdir():
        if file.suffix == ".pt":
            file.unlink()

    for command in tasks[arguments.task]:
        os.system(f"python main.py play --train 1 {command}")
