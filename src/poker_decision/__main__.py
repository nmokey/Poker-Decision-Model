"""One project entry point; optional dependencies load only for the selected task."""

import argparse
import sys
from importlib import import_module

COMMANDS = {
    "prepare": ("poker_decision.pipeline", "Audit PokerBench, save splits and CPU baselines"),
    "train": ("poker_decision.training", "Fit a decision scorer on the prepared holdout"),
    "evaluate": ("poker_decision.experiments", "Run the scenario-grouped backbone/head matrix"),
    "opponents": (
        "poker_decision.opponents.experiment",
        "Generate, select, audit or score forecasts",
    ),
    "holdem": (
        "poker_decision.holdem_integration",
        "Compare Pluribus forecasts and decision models with opponent history",
    ),
    "holdem-scale": (
        "poker_decision.holdem_scaling",
        "Compare nested Hold'em training samples on fixed validation decisions",
    ),
    "holdem-ablate": (
        "poker_decision.holdem_ablations",
        "Run matched GPU model, history, format, duration and seed ablations",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task",
        choices=COMMANDS,
        help="; ".join(f"{key}: {description}" for key, (_, description) in COMMANDS.items()),
    )
    args = parser.parse_args(sys.argv[1:2])
    remaining = sys.argv[2:]
    module = COMMANDS[args.task][0]
    sys.argv = [f"{parser.prog} {args.task}", *remaining]
    import_module(module).main()


if __name__ == "__main__":
    main()
