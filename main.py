"""Master script: run all experiments E1-E8 and save figures.

Usage: python main.py [--quick]
  --quick: use smaller problem sizes for fast testing
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description='Run all OT experiments')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: smaller problems for testing')
    parser.add_argument('--only', type=str, default=None,
                        help='Run only specific experiments, e.g. "e1,e2,e5"')
    args = parser.parse_args()

    experiments = ['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8']

    if args.only:
        experiments = [e.strip().lower() for e in args.only.split(',')]

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Entropy-Regularized Optimal Transport — All Experiments ║")
    print("║  Sinkhorn, Dual Gradient, Log-Sinkhorn, ADMM, ε-Scaling ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    if args.quick:
        print("  [QUICK MODE] Using reduced problem sizes\n")

    total_start = time.time()

    for exp in experiments:
        try:
            if exp == 'e1':
                from experiments.run_e1 import run
                run()
            elif exp == 'e2':
                from experiments.run_e2 import run
                run()
            elif exp == 'e3':
                from experiments.run_e3 import run
                run()
            elif exp == 'e4':
                from experiments.run_e4 import run
                run()
            elif exp == 'e5':
                from experiments.run_e5 import run
                run()
            elif exp == 'e6':
                from experiments.run_e6 import run
                run()
            elif exp == 'e7':
                from experiments.run_e7 import run
                run()
            elif exp == 'e8':
                from experiments.run_e8 import run
                run()
            else:
                print(f"  Unknown experiment: {exp}")
        except Exception as e:
            print(f"  [ERROR] {exp} failed: {e}")
            import traceback
            traceback.print_exc()

    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"All experiments completed in {total_time:.1f}s")
    print(f"Figures saved in: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')}")


if __name__ == '__main__':
    main()
