"""Download external project inputs into their original configured paths.

The registry in ``data/external_sources.json`` pins source revisions and
checksums. Locally generated datasets are deliberately outside this downloader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "data" / "external_sources.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def oolel_content_sha256(frame: pd.DataFrame, columns: list[str]) -> str:
    payload = (
        frame[columns]
        .fillna("")
        .astype(str)
        .to_csv(index=False, lineterminator="\n")
        .encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()


def load_registry() -> dict[str, dict[str, object]]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def target_path(specification: dict[str, object]) -> Path:
    return ROOT / str(specification["target_path"])


def verify(name: str, specification: dict[str, object]) -> None:
    destination = target_path(specification)
    if not destination.exists():
        raise FileNotFoundError(f"{name}: missing {destination}")

    if specification["type"] == "huggingface_dataset":
        frame = pd.read_parquet(destination)
        expected_columns = list(specification["columns"])
        if list(frame.columns) != expected_columns:
            raise ValueError(
                f"{name}: columns {list(frame.columns)} != {expected_columns}"
            )
        if len(frame) != int(specification["rows"]):
            raise ValueError(f"{name}: rows {len(frame)} != {specification['rows']}")
        actual = oolel_content_sha256(frame, expected_columns)
        expected = str(specification["content_sha256"])
    else:
        actual = sha256_file(destination)
        expected = str(specification["sha256"])

    if actual.casefold() != expected.casefold():
        raise ValueError(f"{name}: checksum {actual} != expected {expected}")
    print(f"[verified] {name}: {destination.relative_to(ROOT)}")


def _download_huggingface_file(specification: dict[str, object], destination: Path) -> None:
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(
        repo_id=str(specification["repository"]),
        repo_type="dataset",
        revision=str(specification["revision"]),
        filename=str(specification["source_path"]),
    )
    shutil.copyfile(cached, destination)


def _download_huggingface_dataset(
    specification: dict[str, object], destination: Path
) -> None:
    from datasets import load_dataset

    dataset = load_dataset(
        str(specification["repository"]),
        revision=str(specification["revision"]),
        split=str(specification["split"]),
    )
    frame = dataset.to_pandas()
    expected = list(specification["columns"])
    # The source card has used both names. Preserve the project schema.
    if "non_standard" in frame.columns and "non_standardized" not in frame.columns:
        frame = frame.rename(columns={"non_standard": "non_standardized"})
    frame = frame[expected]
    frame.to_parquet(destination, index=False)


def _download_http_file(specification: dict[str, object], destination: Path) -> None:
    request = urllib.request.Request(
        str(specification["url"]),
        headers={"User-Agent": "wolof-normalization-reproduction/1.0"},
    )
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def _download_git_file(specification: dict[str, object], destination: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="wolof-external-") as temporary:
        checkout = Path(temporary) / "repository"
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--filter=blob:none",
                "--no-checkout",
                str(specification["repository"]),
                str(checkout),
            ],
            check=True,
        )
        with destination.open("wb") as output:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(checkout),
                    "show",
                    f"{specification['revision']}:{specification['source_path']}",
                ],
                check=True,
                stdout=output,
            )


DOWNLOADERS = {
    "huggingface_file": _download_huggingface_file,
    "huggingface_dataset": _download_huggingface_dataset,
    "http_file": _download_http_file,
    "git_file": _download_git_file,
}


def download(name: str, specification: dict[str, object], *, force: bool) -> None:
    destination = target_path(specification)
    if destination.exists() and not force:
        verify(name, specification)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.download")
    temporary.unlink(missing_ok=True)
    try:
        DOWNLOADERS[str(specification["type"])](specification, temporary)
        temporary.replace(destination)
        verify(name, specification)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    registry = load_registry()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "names",
        nargs="*",
        metavar="SOURCE",
        help=f"one of: {', '.join(sorted(registry))}",
    )
    parser.add_argument("--all", action="store_true", help="process every source")
    parser.add_argument("--force", action="store_true", help="redownload existing files")
    parser.add_argument(
        "--verify-only", action="store_true", help="verify local files without downloading"
    )
    arguments = parser.parse_args()

    names = list(registry) if arguments.all else arguments.names
    if not names:
        parser.error("select one or more source names, or pass --all")
    unknown = sorted(set(names) - set(registry))
    if unknown:
        parser.error(f"unknown source(s): {', '.join(unknown)}")

    for name in names:
        specification = registry[name]
        if arguments.verify_only:
            verify(name, specification)
        else:
            download(name, specification, force=arguments.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
