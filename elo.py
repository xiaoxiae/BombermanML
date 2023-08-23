import os
import json
from random import choice
import argparse

base_agents = [
    "coin_collector_agent",
    "peaceful_agent",
    "random_agent",
    "rule_based_agent",
]

GAMES_FILE = "elo/elo.log"
STATS_FILE = "elo/stats.json"


def play_games(games_to_play: dict):
    """Play games based on the dictionary of agent pair : games to play."""
    if os.path.exists(GAMES_FILE):
        os.remove(GAMES_FILE)

    while games_to_play:
        agents = choice(list(games_to_play))

        games_to_play[agents] -= 1
        if games_to_play[agents] == 0:
            del games_to_play[agents]

        os.system(f"python3 main.py play --agents {agents[0]} {agents[1]} --n-rounds 1 --no-gui")


def load_results():
    results = []
    with open(GAMES_FILE) as f:
        for line in f.read().splitlines():
            results.append(line.split())
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-n",
                        help="Number of games to play. Defaults to 1000.", action='store_const', default=1000)
    parser.add_argument("-k",
                        help="The k-factor for calculating the elo. Defaults to 32.", action='store_const', default=32)

    subparsers = parser.add_subparsers(dest="mode")

    agent_subparser = subparsers.add_parser("agent",
                                            help='Benchmarks the agent against the currently added agents, adding it to the system.')

    agent_subparser.add_argument("agent", help="The name of the agent to benchmark.")

    base_subparser = subparsers.add_parser("base",
                                           help="Generates the elos for the base agents.")

    arguments = parser.parse_args()
    games_to_play = {}

    if arguments.mode == 'base':
        games_to_play = {}

        for i in range(len(base_agents)):
            for j in range(i + 1, len(base_agents)):
                games_to_play[(base_agents[i], base_agents[j])] = arguments.n

    elif arguments.mode == 'agent':
        # TODO: also add custom agents from reading the json stats file
        for i in range(len(base_agents)):
            games_to_play[(arguments.agent, base_agents[i])] = arguments.n

    play_games(games_to_play)
    results = load_results()

    if arguments.mode == 'base':
        stats = {
            "base": {agent: 1000 for agent in base_agents},
            "base_games": results
        }
    elif arguments.mode == 'agent':
        stats = json.load(open(STATS_FILE))
        benchmark_agent_elo = 1000

    for a1, result, a2 in results:
        # https://www.omnicalculator.com/sports/elo
        a = stats["base"][a1] if a1 in stats["base"] else stats["other"]
        b = stats["base"][a2] if a2 in stats["base"] else stats["other"]
        score = 1 if result == ">" else 0.5 if result == "=" else 0

        expected = 1 / (10 ** ((b - a) / 400) + 1)
        difference = arguments.k * (score - expected)

        if arguments.mode == 'base':
            stats["base"][a1] += difference
            stats["base"][a2] -= difference

        elif arguments.mode == 'agent':
            benchmark_agent_elo += difference

    if arguments.mode == 'agent':
        if "other" not in stats:
            stats["other"] = {}

        stats["other"][arguments.agent] = benchmark_agent_elo

        if "other_games" not in stats:
            stats["other_games"] = {}

        stats["other_games"][arguments.agent] = results

    json.dump(stats, open(STATS_FILE, 'w'))
