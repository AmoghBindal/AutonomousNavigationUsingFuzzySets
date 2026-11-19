"""Refresh static charts, results and GIFs together after editing scenarios.py."""
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-only", action="store_true",
                        help="Refresh all PNG charts and results, keeping existing GIFs.")
    args = parser.parse_args()
    for module in ("experiments.make_figures", "experiments.make_paper_figures", "experiments.make_animations"):
        command = [sys.executable, "-B", "-m", module]
        if module == "experiments.make_animations" and args.static_only:
            command.append("--static-only")
        print("running", " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
