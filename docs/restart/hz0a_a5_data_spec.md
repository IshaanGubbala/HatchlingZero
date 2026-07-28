# HZ-0A A5 Data Pipeline Specification

Date: July 28, 2026

## Purpose

This document defines the restart-era contract for the HZ-0A data pipeline. It starts from deterministic local manifests and token packing, then expands toward the larger staged mixture required by the plan.

## Core Requirements

The pipeline must support:

- document ingestion
- provenance tracking
- license metadata
- deterministic splits
- tokenizer application
- packed-sequence generation
- deterministic shuffling
- resumable iteration
- dataset manifests and hashes

## Initial Restart Scope

Because the full external training corpus is not yet restored, the restart begins with local-manifest-driven source data.

Current input class:

- repository docs
- restart specs
- archived reports
- reference code
- JSON/config files

This is not the final HZ-0A training mixture. It is the deterministic substrate used to rebuild the pipeline itself.

## Split Contract

Every source record must eventually carry:

- path
- category
- license label
- provenance label
- split assignment
- source SHA-256
- content SHA-256

Allowed splits:

- `train`
- `validation`
- `test`

## Token Packing Contract

Packed batches must record:

- tokenizer SHA-256
- source manifest SHA-256
- sequence length
- document order policy
- total packed sequences
- total tokens

## Restart Deliverables

- source manifest template
- source-manifest audit script
- token-packing script
- packing audit JSON

These are the first A5 artifacts being added now.

`scripts/hz0a_ingest_local_sources.py` now provides deterministic local ingestion for approved text suffixes. It records source/content hashes, category, internal license/provenance labels, and hash-derived train/validation/test splits, while excluding virtual environments, build outputs, dependency trees, and binary files by default. This expands the pipeline contract without claiming that the resulting local corpus is the final external HZ-0A mixture.

The source-manifest audit now uses an inverted normalized-five-token-shingle index to generate near-duplicate candidates, followed by exact Jaccard verification. This preserves the near-duplicate and cross-split contamination checks while avoiding the previous all-pairs comparison on large manifests.

`scripts/hz0a_prepare_wikitext.py` streams archived Wikitext-103 JSONL, extracts the `text` field, preserves train/validation/test boundaries, and emits normalized text plus source/content hashes. This avoids packing JSON serialization and provides a reproducible external-corpus input for the 100M-token gate.

`scripts/hz0a_count_tokens_streaming.py` counts tokenizer output in bounded batches, avoiding a full-corpus token array and making the stage budget measurable on large corpora.
