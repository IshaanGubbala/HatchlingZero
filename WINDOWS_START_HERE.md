# Windows / RTX 3060: start here

LAN relay (`~/hz0a_transfer/`) is currently unreachable from this machine's
side (server up, correct IP/port, firewall open — the block is somewhere on
the network path, still being diagnosed). Don't wait on it. `git pull` on
`main` instead; everything needed is already committed.

**Instructions**: [`docs/rtx3060_g1_matched_transformer.md`](docs/rtx3060_g1_matched_transformer.md)

That doc covers: why the existing matched-Transformer control doesn't
validly compare against the live `G1` run (different corpus, different
sequence length, different eval-set size — not just different token counts),
which data files to copy over manually (`data/` is gitignored, not in
`git pull`), the smoke test, the real-run command, and how to report
results back once the relay is reachable again (or via any other channel
in the meantime).

Background on this machine generally, if this is a cold start:
[`docs/rtx3060_windows_setup.md`](docs/rtx3060_windows_setup.md).
