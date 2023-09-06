import argparse
import json
import os
import subprocess
from random import shuffle

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

    for (a1, a2), games in games_to_play.items():
        print(f"Playing {games} games between {a1} and {a2}...")

        subprocess.Popen(
            ["python3", "main.py", "play", "--agents", a1, a2, "--n-rounds", str(games), "--no-gui"],
            stdout=subprocess.DEVNULL
        ).communicate()


def load_results():
    results = []
    with open(GAMES_FILE) as f:
        for line in f.read().splitlines():
            results.append(line.split())
    return results


def print_stats():
    stats = json.load(open(STATS_FILE))

    print("Base agents:")
    print(json.dumps(stats['base'], indent=4))

    if 'other' in stats:
        print()
        print("Other agents:")
        print(json.dumps(stats['other'], indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-n",
                        help="Number of games to play. Defaults to 100.", action='store_const', default=100)
    parser.add_argument("-k",
                        help="The k-factor for calculating the elo. Defaults to 10.", action='store_const', default=10)
    parser.add_argument("-b", "--base-elo",
                        help="Base elo for calculating stuff. Defaults to 1000.", action='store_const', default=1000)

    subparsers = parser.add_subparsers(dest="mode", required=True)

    agent_subparser = subparsers.add_parser("agent",
                                            help='Benchmarks the agent against the currently added agents, adding it to the system.')

    agent_subparser.add_argument("agent", help="The name of the agent to benchmark.")

    base_subparser = subparsers.add_parser("base",
                                           help="Generates the elos for the base agents.")

    base_subparser.add_argument("--recalculate", help="Recalculate the elos only, don't play the games."
                                                      "Useful for changing the k-factor.",
                                action='store_true')

    stats_subparser = subparsers.add_parser("stats",
                                            help="Prints statistics about the current elos of agents.")

    arguments = parser.parse_args()

    if arguments.mode == 'stats':
        print_stats()
        quit()

    # first, determine what games we need to play
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

    # then actually play them (or just recalculate if --recalculate is specified)
    if arguments.mode == 'agent' or (arguments.mode == 'base' and not arguments.recalculate):
        play_games(games_to_play)
        results = load_results()

        # we have to shuffle the results because the order of the games matters quite a bit for calculating elo
        # this way it somewhat randomizes it, and while it isn't ideal, it's much better than not sorting
        shuffle(results)
    else:
        results = json.load(open(STATS_FILE))["base_games"]

    # then we create the stats dictionary for base, or load it for agent
    if arguments.mode == 'base':
        stats = {
            "base": {agent: arguments.base_elo for agent in base_agents},
            "base_games": results
        }
    elif arguments.mode == 'agent':
        stats = json.load(open(STATS_FILE))

        if "other" not in stats:
            stats["other"] = {}

        if "other_games" not in stats:
            stats["other_games"] = {}

        stats["other"][arguments.agent] = arguments.base_elo
        stats["other_games"][arguments.agent] = results

    # the actual elo calculation
    for a1, result, a2 in results:
        a = stats["base"][a1] if a1 in stats["base"] else stats["other"][a1]
        b = stats["base"][a2] if a2 in stats["base"] else stats["other"][a2]
        score = 1 if result == ">" else 0.5 if result == "=" else 0

        expected = 1 / (10 ** ((b - a) / 400) + 1)
        difference = arguments.k * (score - expected)

        if arguments.mode == 'base':
            stats["base"][a1] += difference
            stats["base"][a2] -= difference

        elif arguments.mode == 'agent':
            stats["other"][a1] += difference

    json.dump(stats, open(STATS_FILE, 'w'))

    print_stats()
