from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path

from core.config import Settings
from core.runner import EvaluationJob, EvaluationRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the competition pipeline locally")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-id", default="local-evaluation")
    parser.add_argument(
        "--pipeline-factory",
        default=os.environ.get("COMPETITION_PIPELINE_FACTORY"),
        help="module:function used to build the pipeline (or set COMPETITION_PIPELINE_FACTORY)",
    )
    args = parser.parse_args()

    output_root = args.output.resolve()
    settings = Settings(
        workspace=output_root.parent,
        answer_root=output_root,
        log_root=output_root.parent / "logs",
        callback_url=None,
        pipeline_factory=args.pipeline_factory,
    )
    runner = EvaluationRunner(settings)
    result = runner.run(
        EvaluationJob(
            request_id=f"local-{uuid.uuid4()}",
            evaluation_id=str(args.evaluation_id),
            dataset_path=args.dataset.resolve(),
        ),
        send_callback=False,
    )
    print(result)


if __name__ == "__main__":
    main()
