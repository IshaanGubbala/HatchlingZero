# HZ-0A A5 Data Audit

Date: July 28, 2026

## Status

The first deterministic A5 data-pipeline artifacts now exist and execute against the rebuilt A4 tokenizer.

The archived Wikitext-103 source has now been normalized and counted with the bounded tokenizer counter. The reproducible budget report is `docs/restart/hz0a_wikitext_token_budget.json`: `196,028,717` total tokens, including `195,148,063` train tokens and separate validation/test splits. The raw 100M-token availability sub-gate is therefore satisfied; packed-output reconstruction and large-scale resumable iteration remain to be verified.

The streaming packer and `StreamingResumablePackedDataset` now provide the large-corpus path. The protocol-aligned train pack produced `190,574` length-1024 JSONL sequences (`195,147,776` packed tokens, `287` tail tokens), and the offset-indexed cursor has exact snapshot/resume regression coverage. The length-128 pack remains a smaller smoke artifact.

The pipeline now validates required provenance/license/split fields and source existence, reports exact-content duplicate groups, and records a seeded document order. Token packing uses a stable path sort followed by a seeded shuffle and records that policy in its audit.

The manifest audit now also computes deterministic normalized five-token-shingle Jaccard similarity and reports near-duplicate pairs at a configurable threshold. The current eight-record manifest has zero exact or near-duplicate groups at threshold `0.9`.

Cross-split contamination is reported separately for exact and near-duplicate pairs. The current manifest has zero contamination groups across train, validation, and test.

`restart/hz0a_dataset.py` now provides a resumable packed-sequence iterator with seeded permutation order, data-pass accounting, and JSON-serializable cursor snapshots. Resume tests prove the next batches are identical after restoring a snapshot.

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
- packed iteration can resume exactly from a serialized cursor
- exact and normalized-shingle near-duplicate groups are reported
- cross-split exact/near-duplicate contamination is reported

## What This Does Not Yet Prove

- full external training-mixture reconstruction
- large-corpus deduplication/removal policy and contamination checks
- large-corpus contamination enforcement and removal policy
- large-scale resumable iteration beyond the current local scaffold
- the full staged mixture percentages beyond the archived Wikitext source
- external-license coverage beyond the recorded Wikitext/internal labels

## A5 Current Assessment

A5's 100M-token reconstruction sub-gate is now satisfied: the Wikitext source has a measured 196M-token budget, the train split has a deterministic 195M-token packed output, and the streaming cursor resumes exact batches. A5 remains open only for the broader staged mixture and external-source policy beyond this reproducible corpus.
