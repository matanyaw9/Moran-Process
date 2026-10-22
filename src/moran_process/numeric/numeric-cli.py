#!/usr/bin/python3

import argparse
import json
import os
import subprocess


def complete(size: int) -> list[list[int]]:
    return [[j for j in range(size) if j != i] for i in range(size)]


def cycle(size: int) -> list[list[int]]:
    return [[(i - 1) % size, (i + 1) % size] for i in range(size)]


def star(size: int) -> list[list[int]]:
    return [[i for i in range(1, size)]] + [[0]] * (size - 1)


def tree(size: int) -> list[list[int]]:
    return [
        [j for j in [(i - 1) // 2, 2 * i + 1, 2 * i + 2] if 0 <= j < size]
        for i in range(size)
    ]


shape_map = {
    "complete": complete,
    "cycle": cycle,
    "star": star,
    "tree": tree,
}


def main():
    parser = argparse.ArgumentParser(prog="numeric-cli")
    parser.add_argument("shape", choices=shape_map.keys(), help="grpah's shape")
    parser.add_argument("size", type=int, help="graph's size")
    parser.add_argument("r", type=float, help="mutant fitness")
    parser.add_argument("-j", type=int, help="number of threads to use")
    args = parser.parse_args()
    if not (0 < args.size < 64):
        parser.error("graph size must be in range [1, 63]")
    if not (args.r >= 0.0):
        parser.error("mutant fitness cannot be negative")

    dir_path = os.path.dirname(__file__)
    exe_path = dir_path + "/numeric-solve"
    if not os.path.isfile(exe_path):
        subprocess.run(
            [
                "rustc",
                "-O",
                "-C",
                "panic=abort",
                "--edition",
                "2024",
                "-o",
                exe_path,
                dir_path + "/src/main.rs",
            ],
            check=True,
        )

    input = {
        "graph": shape_map[args.shape](args.size),
        "fitness": args.r,
    }
    if args.j is not None:
        input |= {"threads": args.j}
    input = json.dumps(input).encode()
    completed = subprocess.run(
        [
            exe_path,
        ],
        input=input,
        check=True,
        capture_output=True,
    )
    output = json.loads(completed.stdout.decode())
    print("prob            time            ctime")
    for line in output:
        print(f"{line["prob"]: <16}{line["time"]: <16}{line["ctime"]}")


if __name__ == "__main__":
    main()
