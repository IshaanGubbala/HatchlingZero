# HZ-0A A5 Data Audit

Date: July 28, 2026

## Status

The first deterministic A5 data-pipeline artifacts now exist and execute against the rebuilt A4 tokenizer.

The pipeline now validates required provenance/license/split fields and source existence, reports exact-content duplicate groups, and records a seeded document order. Token packing uses a stable path sort followed by a seeded shuffle and records that policy in its audit.

## Source Artifacts

- Source manifest:
  - `/Users/ishaangubbala/Documents/Training/data/hz0a_source_manifest.json`
- Source-manifest audit:
  - `/Users/ishaangubbala/Documents/Training/data/source_manifest_audit.json`
- Token packer:
  - `/Users/ishaangubbala/Documents/Training/scripts/hz0a_pack_tokens.py`
- Packed train split:
  - `/Users/ishaangubbala/Documents/Training/data/packed/train_packed.json`
- Packed train audit:
  - `/Users/ishaangubbala/Documents/Training/data/packed/train_packed.audit.json`

## Verified Current Outputs

From the current audits:

- source manifest SHA-256: `a16166f171beb58eb1342c9f4e8267d6e926ca81ff5689814025b1932a96fb47`
- tokenizer SHA-256 used for packing: `cab29d54ca82f902472996939b9441a7bf3b0bb2e80f89d7f4a8d7445b240eb1`
- train split total input tokens: `16,739`
- packed train sequences at length `128`: `130`
- packed-output SHA-256: `316cc535069d4f6443d0769033aa84904d9bf7085836fb3d1fdb3786c916ef83`

Included train sources currently are:

- `README.md`
- `plans/HZ-0A_Total_Restart_Plan.md`
- `reference/hz0a_gdn2_reference.py`
- `archive/docs/architecture.md`

## What This Proves

- provenance-aware source manifests are now executable
- split-aware audits are now executable
- tokenizer and packer now interoperate deterministically
- the repo can produce reproducible packed sequences from current source data

## What This Does Not Yet Prove

- full external training-mixture reconstruction
- deduplication / near-duplicate removal
- contamination checks
- large-scale resumable iteration
- 100M-token rebuild target from the restart plan

## A5 Current Assessment

A5 is meaningfully in progress, but not complete. The restart now has validated deterministic source manifests, duplicate reporting, seeded document ordering, reproducible packed-sequence generation, and public-script regression tests. It remains far below the plan's 100M-token reconstruction gate and still needs near-duplicate removal, contamination checks, resumable large-scale iteration, and the full staged mixture.
