import argparse


def parseArgs():
    parser = argparse.ArgumentParser(
        description="Run the simulation with the arguments."
    )
    parser.add_argument(
        "-f",
        "--file",
        default="vle-0.80-reference.json",
        type=str,
        help="input json file path",
    )
    parser.add_argument(
        "--log",
        default="logs/log1.log",
        type=str,
        help="input log file path",
    )
    args = parser.parse_args()
    return args
