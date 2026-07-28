"""Build a deterministic provenance manifest from approved local text sources."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TEXT_SUFFIXES = {".c", ".cpp", ".h", ".hpp", ".json", ".md", ".py", ".rs", ".sh", ".swift", ".toml", ".txt", ".yaml", ".yml"}
DEFAULT_EXCLUDES = {".git", ".venv", ".venv-py39", "__pycache__", "node_modules", "target", "build", "dist"}


def category_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".py", ".rs", ".c", ".cpp", ".h", ".hpp", ".sh", ".swift"}:
        return "code"
    if suffix in {".json", ".yaml", ".yml", ".toml"}:
        return "json_and_configuration"
    if "debug" in path.name.lower() or "status" in path.parts:
        return "terminal_and_debugging"
    return "documentation"


def split_for(content_hash: str, validation_fraction: float, test_fraction: float) -> str:
    value = int(content_hash[:16], 16) / float(16 ** 16)
    if value < test_fraction:
        return "test"
    if value < test_fraction + validation_fraction:
        return "validation"
    return "train"


def ingest(roots: list[Path], output: Path, validation_fraction: float = 0.05, test_fraction: float = 0.05, excludes: set[str] | None = None) -> dict:
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must be non-negative and sum to less than one")
    excluded = DEFAULT_EXCLUDES | (excludes or set())
    candidates = []
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES or any(part in excluded for part in path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            candidates.append((path, text))
    records = []
    for path, text in sorted(candidates, key=lambda item: str(item[0])):
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        records.append({"path": str(path), "category": category_for(path), "license": "internal-local-source", "provenance": "local-ingestion", "split": split_for(content_hash, validation_fraction, test_fraction), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "content_sha256": content_hash, "bytes": len(text.encode("utf-8"))})
    payload = {"version": "a5.ingested.v1", "roots": [str(root) for root in roots], "excluded_directory_names": sorted(excluded), "records": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest approved local HZ-0A text sources into a deterministic manifest.")
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    args = parser.parse_args()
    payload = ingest(args.roots, args.output, args.validation_fraction, args.test_fraction)
    print(json.dumps({"output": str(args.output), "record_count": len(payload["records"])}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
