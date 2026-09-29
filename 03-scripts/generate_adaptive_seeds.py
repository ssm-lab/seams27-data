"""Generate Java adaptation seeds once; no server or training is started."""
import argparse
import json
import random
from pathlib import Path


def generate_seeds(master_seed=42, count=10):
    return random.Random(master_seed).sample(range(1, 2**31), count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--master-seed', type=int, default=42)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / '01-configurations/adaptive/experiment-10-seed-42/seeds.json')
    args = parser.parse_args()
    seeds = generate_seeds(args.master_seed, args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'master_seed': args.master_seed, 'seeds': seeds}, indent=2) + '\n')
    for index, seed in enumerate(seeds, 1):
        print(f'Run {index}: seed={seed}')


if __name__ == '__main__':
    main()
