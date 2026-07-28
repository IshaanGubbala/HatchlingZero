# Python Side

This directory is reserved for thin Python helpers used during restart
validation, such as:

- numerical comparison scripts
- checkpoint replay harnesses
- reference-vs-fused diff tools

The restart should avoid rebuilding a full Python-first training stack here.
Python should validate and orchestrate; the fused operator path should live in
the Rust workspace.
