from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader


DEFAULT_SOURCES = [
    Path("README.md"),
    Path("docs/architecture.md"),
    Path("docs/hz0a/audit.md"),
    Path("vendor/GatedDeltaNet-2/README.md"),
    Path("/Users/ishaangubbala/.codex/attachments/68bc2a86-a9f7-4dd6-aac6-d1b5874ebde0/pasted-text.txt"),
    Path("/Users/ishaangubbala/Downloads/HATCHLING-ZERO Development Plan.pdf"),
]


def read_source(path: Path) -> str:
    if not path.exists():
        return ""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def build_corpus(paths: list[Path]) -> str:
    sections = []
    for path in paths:
        text = read_source(path).strip()
        if not text:
            continue
        sections.append(f"\n\n### SOURCE: {path}\n\n{text}\n")
    return "\n".join(sections).strip() + "\n"


def split_corpus(text: str, train_ratio: float = 0.9) -> tuple[str, str]:
    pivot = max(1, int(len(text) * train_ratio))
    return text[:pivot], text[pivot:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--source", action="append", default=None)
    args = parser.parse_args()

    sources = [Path(item) for item in args.source] if args.source else DEFAULT_SOURCES
    corpus = build_corpus(sources)
    if not corpus.strip():
        raise RuntimeError("No corpus content could be gathered from the provided sources.")

    train_text, val_text = split_corpus(corpus, train_ratio=args.train_ratio)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "hz0a_seed_train.txt"
    val_path = args.output_dir / "hz0a_seed_val.txt"
    meta_path = args.output_dir / "hz0a_seed_sources.txt"

    train_path.write_text(train_text, encoding="utf-8")
    val_path.write_text(val_text, encoding="utf-8")
    meta_path.write_text("\n".join(str(path) for path in sources), encoding="utf-8")

    print(f"train_path={train_path}")
    print(f"val_path={val_path}")
    print(f"train_chars={len(train_text)}")
    print(f"val_chars={len(val_text)}")


if __name__ == "__main__":
    main()
