"""Single entry point.

  python3 run.py --plan                       show the trial grid and call budget, no calls
  python3 run.py --fake                       full offline run with a fake model (no keys)
  python3 run.py --api                        real run through an OpenAI-compatible API
  python3 run.py --hf                         real run with a local Hugging Face model (CPU)
  python3 run.py --gpu                        same, on CUDA GPUs (implies --hf)
  python3 run.py --gpu --hf-model Qwen/Qwen3.6-27B   choose another Hugging Face model
  python3 run.py --limit 5                    run only the first 5 trials (smoke test)

With none of --fake, --api and --hf, the backend named by llm.backend in config.json is used
(default: api).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mas_sim.llm import resolve_backend  # noqa: E402
from mas_sim.runner import describe_plan, load_config, run  # noqa: E402
from mas_sim.scenarios import verify_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run both MAS experiments and compute all metrics.")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                         "config.json"))
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--fake", action="store_true", help="use the offline fake model")
    backend.add_argument("--api", action="store_true",
                         help="use an OpenAI-compatible API (base_url, api_key, model)")
    backend.add_argument("--hf", action="store_true",
                         help="use a local Hugging Face model (default Qwen/Qwen3.8-27B)")
    parser.add_argument("--gpu", action="store_true",
                        help="run the Hugging Face model on CUDA GPUs; implies --hf")
    parser.add_argument("--hf-model", metavar="NAME",
                        help="Hugging Face model id or local path; implies --hf")
    parser.add_argument("--plan", action="store_true", help="print the grid and budget, then exit")
    parser.add_argument("--limit", type=int, help="run only the first N trials")
    parser.add_argument("--output-dir", help="override the output directory")
    args = parser.parse_args()

    if args.plan:
        verify_all()
        print(json.dumps(describe_plan(load_config(args.config)), indent=2))
        return
    chosen = "api" if args.api else "hf" if args.hf else None
    try:
        resolve_backend(None, args.fake, chosen, args.gpu, args.hf_model)
    except ValueError as error:
        parser.error(str(error))
    run(args.config, fake=args.fake, limit=args.limit, output_dir=args.output_dir,
        backend=chosen, gpu=args.gpu, hf_model=args.hf_model)


if __name__ == "__main__":
    main()
