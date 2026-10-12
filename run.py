"""Single entry point.

  python3 run.py --plan          show the trial grid and call budget, no calls
  python3 run.py --fake          full offline run with a fake model (no keys)
  python3 run.py                 real run using the settings in config.json
  python3 run.py --limit 5       run only the first 5 trials (smoke test)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mas_sim.runner import describe_plan, load_config, run  # noqa: E402
from mas_sim.scenarios import verify_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run both MAS experiments and compute all metrics.")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                         "config.json"))
    parser.add_argument("--fake", action="store_true", help="use the offline fake model")
    parser.add_argument("--plan", action="store_true", help="print the grid and budget, then exit")
    parser.add_argument("--limit", type=int, help="run only the first N trials")
    parser.add_argument("--output-dir", help="override the output directory")
    args = parser.parse_args()

    if args.plan:
        verify_all()
        print(json.dumps(describe_plan(load_config(args.config)), indent=2))
        return
    run(args.config, fake=args.fake, limit=args.limit, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
