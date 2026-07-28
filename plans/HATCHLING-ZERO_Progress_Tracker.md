# HATCHLING-ZERO Master Progress Tracker

Updated: July 28, 2026

## Purpose

This tracker translates the master development plan into restart-era execution status. It is a governance file for the full HATCHLING-ZERO program, not a source of implementation truth.

## Overall Status

- Program state: restart in progress
- Active focus: HZ-0A restart archaeology and specification recovery
- Current branch intent: strip the repo to restart essentials, then rebuild stage-by-stage
- Blocking rule: do not advance dependent stages until predecessor contracts are frozen

## Stage Summary

| Stage | Goal | Status | Notes |
| --- | --- | --- | --- |
| HZ-0A | Rebuild recurrent-hybrid base | In progress | Phase A0 started; recovery docs created |
| HZ-0B | Session-local associative memory | Not started | Must wait for frozen HZ-0A base |
| HZ-0C | Session-local fast weights | Not started | Must wait for stable HZ-0B semantics |
| HZ-0D | Adaptive recurrence / compute | Not started | Can simulate early, integration waits for HZ-0C |
| HZ-0E | Micro-MoE specialization | Not started | Depends on stable HZ-0D |

## Program Rules

- No legacy implementation is trusted without re-derivation or reproduction.
- Historical metrics are evidence, not claims.
- Every stage needs:
  - history audit
  - recovered requirements/spec
  - tiny simulator/reference
  - tests
  - baseline comparisons
  - integration only after isolated validation

## Current Evidence Base

- HZ-0A restart docs now exist:
  - `/Users/ishaangubbala/Documents/Training/docs/restart/hz0a_history_audit.md`
  - `/Users/ishaangubbala/Documents/Training/docs/restart/hz0a_recovered_spec.md`
- The repo has been intentionally reduced, with legacy material moved under `/Users/ishaangubbala/Documents/Training/archive`

## Immediate Next Milestones

1. Freeze the authoritative HZ-0A 300M specification.
2. Build the tiny mathematical GDN-2 reference.
3. Derive and validate backward math before any fused PMetal training work.
4. Only then begin HZ-0A tokenizer/data/training stack rebuild.

## Risks

- Legacy docs contain conflicting confidence levels; overclaims must not leak into the restart.
- The old "110M" versus "292M" confusion can invalidate comparisons if reused carelessly.
- Streaming and memory claims from legacy code are not trusted starting points.

## Stop / Go Gate

- `GO` for HZ-0A A1 specification work
- `STOP` for HZ-0B through HZ-0E implementation work until HZ-0A is frozen
