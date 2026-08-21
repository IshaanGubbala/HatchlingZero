# Agent notes for this repo

Also read `CLAUDE.md` in this same directory -- the multi-machine
dispatch-relay topology (Mac / Pi / Windows RTX3060) it documents applies to
every agent working here, not just Claude Code.

## Running a command on a GPU: `scripts/runpod_run.sh`

This machine (and the Pi) has no CUDA GPU. When a task genuinely needs one
(a real training run, a CUDA-specific benchmark, anything touching
`torch.cuda`) and the Windows RTX3060 relay isn't the right fit, use:

```
scripts/runpod_run.sh -- <command...>
```

It is a plain, agent-agnostic bash script -- read its own header comment for
the full flag reference before using it (`--pull <path>` to copy a result
back, `--keep` to leave the pod up for a follow-up call, `--gpu-id`,
`--ttl-minutes`, `--sync git|local|none`). Do not guess flags from memory;
the script header is the source of truth.

What it does, briefly: reuses an already-running RunPod pod if one exists
(and leaves it running afterward -- it was not created by this call, so it
is not this call's to kill), otherwise creates a cheap one, syncs this repo
to it, runs your command, optionally pulls results back, then terminates
**only the pod it created itself**. A hard `--terminate-after` safety net is
always set on pods it creates, independent of the script's own cleanup.

If `runpodctl pod list` shows a pod already running, say so before assuming
you can use it freely -- it may belong to another concurrent session's job.

Claude Code also has a thin `/runpod-run` skill
(`.claude/skills/runpod-run/SKILL.md`) that just calls this same script.
