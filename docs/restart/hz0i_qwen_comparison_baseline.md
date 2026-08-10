# HZ-0I Qwen comparison baseline

The external target selected for the first gate is `Qwen/Qwen3-0.6B` (596.05M
parameters), the closest public Qwen model to the desired 0.8B reference. A
pretrained evaluation on 16 sequences of 256 Qwen-tokenized tokens from
`data/tokenizer_corpus/all.txt` measured mean CE `3.7882` and perplexity `44.18`.

This is a baseline only. It is not yet a fair BDH comparison because BDH is
not pretrained to the same token budget and the tokenizers differ. The eventual
0.3B BDH gate must report both native-tokenizer CE and a common raw-text
character/byte or re-tokenized evaluation, plus matched training tokens, active
FLOPs, memory, and decode speed.
