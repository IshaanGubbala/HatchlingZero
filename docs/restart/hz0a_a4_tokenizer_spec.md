# HZ-0A A4 Tokenizer Specification

Date: July 28, 2026

## Purpose

This document freezes the restart-era tokenizer contract for HZ-0A Phase A4. It is derived from the archived corpus-prep and tokenizer-training scripts, but it does not assume their outputs are usable.

## Recovered Historical Signals

Evidence from archive:

- `/Users/ishaangubbala/Documents/Training/archive/scripts/train_tokenizer.py`
- `/Users/ishaangubbala/Documents/Training/archive/scripts/prepare_tokenizer_corpus.py`
- `/Users/ishaangubbala/Documents/Training/archive/data/tokenizer/hz_24k.json`

Recovered intent:

- tokenizer family: byte-level BPE
- implementation family: Hugging Face `tokenizers`
- target size: around 24K
- explicit tool/chat/code special tokens
- mixed-domain corpus including prose, code, docs, JSON/tool schemas, and debugging-style text

## Locked A4 Contract

- tokenizer algorithm: byte-level BPE
- vocabulary size: `24,576` total tokens
- pre-tokenizer: byte-level with prefix-space behavior
- post-processor: byte-level trim-offset behavior
- deterministic training: required
- corpus manifest: required
- tokenizer hash: required

## Special Tokens

The restart tokenizer reserves these tokens explicitly:

1. `<|bos|>`
2. `<|eos|>`
3. `<|pad|>`
4. `<|system|>`
5. `<|user|>`
6. `<|assistant|>`
7. `<|tool_list|>`
8. `<|tool_call|>`
9. `<|tool_result|>`
10. `<|tool_error|>`
11. `<|fim_prefix|>`
12. `<|fim_suffix|>`
13. `<|fim_middle|>`
14. `<|code_start|>`
15. `<|code_end|>`

This special-token table is part of the tokenizer hash contract.

## Coverage Requirements

The tokenizer rebuild must cover:

- natural language prose
- code syntax
- shell / terminal text
- JSON and configuration syntax
- API and documentation text
- tool-schema style structured data

## Validation Requirements

The final tokenizer must pass:

- encode/decode round-trip tests
- whitespace preservation tests
- unknown-token behavior checks
- JSON/token-schema coverage probes
- code-snippet tokenization sanity checks

## Implementation Notes

- The environment currently does not have the `tokenizers` package installed, so the deterministic training path must be implemented as repo tooling but cannot be fully executed yet in this shell.
- The A4 rebuild should prefer one canonical tokenizer path rather than mixed byte-level and BPE fallbacks.

## A4 Deliverables

- tokenizer training script
- tokenizer corpus manifest
- tokenizer model files
- tokenizer audit report

The first two are being rebuilt now as deterministic source artifacts.
