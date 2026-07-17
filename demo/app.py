"""Minimal local runner for the verified default XYZT pipeline.

This is project infrastructure, not a notebook-derived competition interface.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run_xyzt_awesome_pipeline, write_xyzt_awesome_submission


def main() -> None:
    """Generate an XYZT submission from local competition CSV files."""
    parser = argparse.ArgumentParser(description="Run the XYZT demand-forecasting pipeline.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output", default="outputs/submission.csv")
    args = parser.parse_args()

    submission = run_xyzt_awesome_pipeline(args.data_dir)
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_xyzt_awesome_submission(submission, output_path)
    print(f"Wrote {len(submission)} predictions to {output_path}")


if __name__ == "__main__":
    main()
