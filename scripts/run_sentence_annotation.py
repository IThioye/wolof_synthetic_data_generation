"""Launch the optimized sentence annotator in an isolated campaign.

Example:
    python scripts/run_sentence_annotation.py --campaign gold_extension_v1 --target 1000
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="gold_extension_v1")
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--distance-threshold", type=float, default=0.10)
    parser.add_argument("--candidate-margin", type=float, default=0.08)
    parser.add_argument("--french-margin", type=float, default=0.04)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args()


def configure_campaign(
    name: str,
    target: int,
    distance_threshold: float = 0.10,
    candidate_margin: float = 0.08,
    french_margin: float = 0.04,
) -> Path:
    if not CAMPAIGN_NAME.fullmatch(name):
        raise ValueError(
            "Campaign names may contain only letters, digits, underscores, and hyphens"
        )
    if target <= 0:
        raise ValueError("The paired-data target must be positive")
    for label, value in (
        ("distance threshold", distance_threshold),
        ("candidate margin", candidate_margin),
        ("French margin", french_margin),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"The {label} must be between zero and one")

    campaign_dir = PROJECT_ROOT / "data" / "annotations" / "campaigns" / name
    os.environ["PFE_GOLD_ANNOTATIONS_PATH"] = str(
        campaign_dir / "gold_annotations.csv"
    )
    os.environ["PFE_TOKEN_CORRECTIONS_PATH"] = str(
        campaign_dir / "token_corrections.csv"
    )
    os.environ["PFE_SENTENCE_ANNOTATION_EVENTS_PATH"] = str(
        campaign_dir / "annotation_events.csv"
    )
    os.environ["PFE_ANNOTATION_TARGET_KEPT"] = str(target)
    os.environ["PFE_ANNOTATION_MAX_DISTANCE_RATIO"] = str(distance_threshold)
    os.environ["PFE_ANNOTATION_MIN_CANDIDATE_MARGIN"] = str(candidate_margin)
    os.environ["PFE_ANNOTATION_FRENCH_AMBIGUITY_MARGIN"] = str(french_margin)
    return campaign_dir


def main() -> None:
    args = parse_args()
    campaign_dir = configure_campaign(
        args.campaign,
        args.target,
        args.distance_threshold,
        args.candidate_margin,
        args.french_margin,
    )
    sys.path.insert(0, str(PROJECT_ROOT))

    # Import only after setting the environment-backed output paths.
    from src.annotation.flask_annotation_app import app

    print(f"Campaign: {args.campaign}")
    print(f"Append-only outputs: {campaign_dir.resolve()}")
    print(f"Paired-data display target: {args.target}")
    print(
        "Lookup gates: "
        f"distance<={args.distance_threshold}, "
        f"candidate margin>={args.candidate_margin}, "
        f"French margin={args.french_margin}"
    )
    print("Frozen benchmark splits are not modified by this application.")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
