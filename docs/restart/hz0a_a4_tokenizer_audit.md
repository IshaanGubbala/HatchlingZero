# HZ-0A A4 Tokenizer Audit

Date: July 28, 2026

## Status

The A4 tokenizer rebuild is now executable in the current environment and produces concrete artifacts from a deterministic corpus manifest.

## Inputs

- Corpus manifest:
  - `/Users/ishaangubbala/Documents/Training/data/tokenizer_corpus_manifest.json`
- Corpus builder:
  - `/Users/ishaangubbala/Documents/Training/scripts/hz0a_prepare_tokenizer_corpus.py`
- Tokenizer trainer:
  - `/Users/ishaangubbala/Documents/Training/scripts/hz0a_train_tokenizer.py`
- Runtime wrapper:
  - `/Users/ishaangubbala/Documents/Training/tokenizer/hz0a_tokenizer.py`

## Produced Artifacts

- Corpus text:
  - `/Users/ishaangubbala/Documents/Training/data/tokenizer_corpus/all.txt`
- Corpus audit:
  - `/Users/ishaangubbala/Documents/Training/data/tokenizer_corpus/audit.json`
- Tokenizer model:
  - `/Users/ishaangubbala/Documents/Training/data/tokenizer/hz0a_24576.json`
- Tokenizer audit:
  - `/Users/ishaangubbala/Documents/Training/data/tokenizer/audit.json`

## Verified Results

From the current tokenizer audit:

- vocabulary size: `24,576`
- corpus SHA-256: `337b26b0466006c4e564aa4f41dbff9ebe81e07b8dd1777345aa40bfcf030f2a`
- tokenizer SHA-256: `cab29d54ca82f902472996939b9441a7bf3b0bb2e80f89d7f4a8d7445b240eb1`
- runtime round-trip status: `all_roundtrip_match = true`

## Important Note About Byte-Level Prefix Space

Raw byte-level decode with `add_prefix_space=true` prepends a leading space for non-space-prefixed strings. This is expected byte-level behavior, not corruption.

The HZ-0A restart therefore treats the runtime wrapper as part of the tokenizer contract:

- raw tokenizer JSON is the model artifact
- `HZ0ATokenizer` is the canonical encode/decode interface for round-trip tests

This preserves:

- deterministic training
- byte-level BPE behavior
- exact runtime round-trip for code, JSON, shell text, and whitespace-sensitive strings

## A4 Exit Assessment

Satisfied in the current repo:

- tokenizer training script exists and runs
- tokenizer corpus manifest exists and runs
- tokenizer model file is produced
- tokenizer audit report is produced
- round-trip validation passes through the canonical runtime wrapper

Still limited:

- the corpus is assembled from current repository and archived materials, not yet from the eventual full-scale A5/A9 training mixture
- broader unknown-token and extended coverage probes should expand further as A5 data sources grow
