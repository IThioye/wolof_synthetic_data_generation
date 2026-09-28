"""Validate project text as UTF-8 and flag common stored mojibake sequences."""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGETS = [
    PROJECT_ROOT / "pfe_report_skeleton.md",
    PROJECT_ROOT / "docs",
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "notebooks",
    PROJECT_ROOT / "data" / "annotations",
    PROJECT_ROOT / "data" / "youtube" / "clean_data.csv",
]
TEXT_SUFFIXES = {
    ".css",
    ".csv",
    ".html",
    ".ipynb",
    ".js",
    ".json",
    ".md",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
}
SKIP_DIRECTORY_NAMES = {
    ".git",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "checkpoints",
    "eflomal",
    "venv",
}
SUSPICIOUS_SEQUENCES = {
    "\u00c3\u00a0": "likely corrupted à",
    "\u00c3\u00a7": "likely corrupted ç",
    "\u00c3\u00a8": "likely corrupted è",
    "\u00c3\u00a9": "likely corrupted é",
    "\u00c3\u00aa": "likely corrupted ê",
    "\u00c3\u00af": "likely corrupted ï",
    "\u00c3\u00b1": "likely corrupted ñ",
    "\u00c3\u00b4": "likely corrupted ô",
    "\u00c3\u00b9": "likely corrupted ù",
    "\u00c2\u00a0": "likely corrupted non-breaking space",
    "\u00e2\u20ac": "likely corrupted punctuation",
    "\u00e2\u201e": "likely corrupted symbol",
    "\u00e2\u201d": "likely corrupted box-drawing character",
    "\u00f0\u0178": "likely corrupted emoji",
    "\ufffd": "Unicode replacement character",
}


def iter_text_files(targets: list[Path]):
    seen = set()
    for target in targets:
        try:
            if target.is_file():
                candidates = [target]
            elif target.is_dir():
                candidates = target.rglob("*")
            else:
                print(f"WARNING missing path: {target}")
                continue
        except OSError as exc:
            print(f"WARNING unreadable path: {target} ({exc})")
            continue

        if not target.exists():
            print(f"WARNING missing path: {target}")
            continue

        for path in candidates:
            if any(part.casefold() in SKIP_DIRECTORY_NAMES for part in path.parts):
                continue
            try:
                is_text_file = path.is_file() and path.suffix.casefold() in TEXT_SUFFIXES
            except OSError as exc:
                print(f"WARNING unreadable path: {path} ({exc})")
                continue
            if not is_text_file:
                continue
            try:
                resolved = path.resolve()
            except OSError as exc:
                print(f"WARNING unresolved path: {path} ({exc})")
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path


def audit_file(path: Path, max_findings: int):
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return [(0, f"invalid UTF-8: {exc}")]

    findings = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        matched = [
            description
            for sequence, description in SUSPICIOUS_SEQUENCES.items()
            if sequence in line
        ]
        control_characters = [
            char for char in line if ord(char) < 32 and char not in {"\t"}
        ]
        if matched or control_characters:
            reason = "; ".join(sorted(set(matched)))
            if control_characters:
                reason += "; unexpected control character"
            findings.append((line_number, reason.strip("; ")))
            if len(findings) >= max_findings:
                break
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--max-findings", type=int, default=20)
    args = parser.parse_args()

    targets = [path.resolve() for path in args.paths] or DEFAULT_TARGETS
    file_count = 0
    issue_count = 0
    for path in iter_text_files(targets):
        file_count += 1
        findings = audit_file(path, args.max_findings)
        for line_number, reason in findings:
            issue_count += 1
            display_path = path.resolve().relative_to(PROJECT_ROOT)
            print(f"{display_path}:{line_number}: {reason}")

    print(f"Audited {file_count:,} files; found {issue_count:,} potential issues.")
    if issue_count:
        print("No files were modified. Review each finding before repairing source data.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
