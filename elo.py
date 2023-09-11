import argparse
import json
import os
import subprocess
import sys
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
    """Play games based on the dictionary of {(agent_1, agent_2): games to play}."""
    if os.path.exists(GAMES_FILE):
        os.remove(GAMES_FILE)

    for (a1, a2), games in games_to_play.items():
        print(f"Playing {games} games between {a1} and {a2}...")

        subprocess.Popen(
            ["python", "main.py", "play", "--agents", a1, a2, "--n-rounds", str(games), "--no-gui"],
            stdout=subprocess.DEVNULL,
        ).communicate()


def load_results():
    results = []
    with open(GAMES_FILE) as f:
        for line in f.read().splitlines():
            results.append(line.split())
    return results


def get_all_games_played(stats):
    games = list(stats['base_games'])

    for a in stats['other_games']:
        games += stats['other_games'][a]

    return games



def print_stats(stats, agent=None):
    def _put_on_left(agent, game):
        """Put the agent on the left side of the game (for easier code)."""
        if game[0] == agent:
            return game
        else:
            return [
                game[2],
                "=" if game[1] == "=" else "<" if game[1] == ">" else ">",
                game[0],
            ]

    print("Base agents:")
    print(json.dumps(stats['base'], indent=4, sort_keys=True))

    if 'other' in stats:
        print()
        print("Other agents:")
        print(json.dumps(stats['other'], indent=4, sort_keys=True))

    if agent is not None:
        print()
        print(f"{agent}'s results against other agents:")

        agent_scores = {}

        for game in get_all_games_played(stats):
            if agent in game:
                _, result, other_agent = _put_on_left(agent, game)

                if other_agent not in agent_scores:
                    agent_scores[other_agent] = [0, 0, 0]  # win / draw / loss

                if result == ">":
                    agent_scores[other_agent][0] += 1
                elif result == "=":
                    agent_scores[other_agent][1] += 1
                else:
                    agent_scores[other_agent][2] += 1

        for other_agent in agent_scores:
            agent_scores[other_agent] = ':'.join(list(map(str, agent_scores[other_agent])))

        print(json.dumps(agent_scores, indent=4, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", help="Number of games. Defaults to 100.", type=int, default=100)

    parser.add_argument("-k",
                        help="The k-factor for calculating the elo. Defaults to 10.", type=int, default=10)

    parser.add_argument("-b", "--base-elo",
                        help="Base elo for calculating stuff. Defaults to 1000.", type=int, default=1000)

    parser.add_argument("--no-save",
                        help="Don't save the result to the file, only calculate and print result.", action='store_true')

    subparsers = parser.add_subparsers(dest="mode", required=True)

    agent_subparser = subparsers.add_parser("agent", add_help=False,
                                            help='Benchmarks the agent against the currently added agents, adding it to the system.')

    agent_subparser.add_argument("agent", help="The name of the agent to benchmark.")

    base_subparser = subparsers.add_parser("base", add_help=False,
                                           help="Generates the elos for the base agents.")

    base_subparser.add_argument("--recalculate", help="Recalculate the elos only, don't play the games."
                                                      "Useful for changing the k-factor.",
                                action='store_true')

    stats_subparser = subparsers.add_parser("stats", add_help=False,
                                            help="Prints statistics about the current elos of agents.")

    stats_subparser.add_argument("--agent", type=str, help="Prints more detailed statistics about one specific agent.")

    duel_subparser = subparsers.add_parser("duel", help="Plays a match between two agents.")

    duel_subparser.add_argument("agent1", help="First duelist")

    duel_subparser.add_argument("agent2", help="Second duelist")

    arguments = parser.parse_args()

    if arguments.mode == 'stats':
        if not os.path.exists(STATS_FILE):
            print("No stats file calculated, run `python elo.py base` first!")
            sys.exit(1)

        print_stats(json.load(open(STATS_FILE)), agent=arguments.agent)
        sys.exit(0)

    # first, determine what games we need to play
    games_to_play = {}

    # for base, it's all pairs
    if arguments.mode == 'base':
        games_to_play = {}

        for i in range(len(base_agents)):
            for j in range(i + 1, len(base_agents)):
                games_to_play[(base_agents[i], base_agents[j])] = arguments.n

    # for agent, it's games against base and also against all other added agents
    elif arguments.mode == 'agent':
        if not os.path.exists(STATS_FILE):
            print("No stats file calculated, run `python elo.py base` first!")
            sys.exit(1)

        stats = json.load(open(STATS_FILE))

        for other_agent in stats.get('other', ()):
            if other_agent == arguments.agent:
                continue

            games_to_play[(arguments.agent, other_agent)] = arguments.n

        for i in range(len(base_agents)):
            games_to_play[(arguments.agent, base_agents[i])] = arguments.n

    # for duel, it's just games against each other
    elif arguments.mode == 'duel':
        games_to_play[(arguments.agent1, arguments.agent2)] = arguments.n

    # then actually play them (or just recalculate if --recalculate is specified)
    if arguments.mode == 'agent' or arguments.mode == 'duel' or (arguments.mode == 'base' and not arguments.recalculate):
        play_games(games_to_play)
        results = load_results()

        # we have to shuffle the results because the order of the games matters quite a bit for calculating elo
        # this way it somewhat randomizes it, and while it isn't ideal, it's much better than not sorting
        shuffle(results)
    else:
        results = json.load(open(STATS_FILE))["base_games"]

    # in duel mode, just print out the result and save nothing
    if arguments.mode == 'duel':
        points = {arguments.agent1: 0, 'equal': 0, arguments.agent2: 0}
        for a1, result, a2 in results:
            res = {'>': a1, '=': 'equal', '<': a2}
            points[res[result]] += 1
        
        print(' | '.join(name + ': ' + str(score) for name, score in points.items()))
        quit()

    # then we create the stats dictionary for base, or load it for agent
    if arguments.mode == 'base':
        stats = {
            "base": {agent: arguments.base_elo for agent in base_agents},
            "base_games": results
        }
    elif arguments.mode == 'agent':
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

    if not arguments.no_save:
        json.dump(stats, open(STATS_FILE, 'w'))

    if arguments.mode == 'agent':
        print_stats(stats, agent=arguments.agent)
    else:
        print_stats(stats)
