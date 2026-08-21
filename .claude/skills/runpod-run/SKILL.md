---
name: runpod-run
description: "Use whenever a command needs a real CUDA GPU this machine doesn't have (training runs, CUDA benchmarks, anything that needs torch.cuda). Runs one shell command on a RunPod GPU pod -- reuses an already-running pod if one exists, otherwise creates one -- and auto-terminates any pod it created when the command finishes, fails, or is interrupted. Trigger: `/runpod-run <command>`, or any request to 'run this on a GPU / on runpod'."
---

# /runpod-run

Thin trigger for `scripts/runpod_run.sh` -- a plain bash script, not
Claude-Code-specific. Any agent in this repo (Codex, a human, a script) can
call it directly; this file just makes it invocable as a slash command.

## Usage

```
/runpod-run <command...>
```

Under the hood this calls:

```
scripts/runpod_run.sh -- <command...>
```

Read `scripts/runpod_run.sh`'s own header comment (top of the file) for the
full flag reference -- `--pull <path>` to rsync a result back after the
command, `--keep` to leave the pod up for a follow-up call, `--gpu-id`,
`--ttl-minutes`, `--sync git|local|none`, etc. Don't restate those flags from
memory; read the file, since it is the single source of truth and this SKILL
file must not drift out of sync with it.

## What it actually does (so you can explain it, not just invoke it)

1. Checks `runpodctl pod list` for an already-running pod. If one exists, it
   is reused -- SSH details come from `runpodctl pod get <id>`.
2. Otherwise creates a new pod (default: single A40, `--terminate-after` set
   as a hard safety net even if this script's own cleanup somehow fails to
   run) and waits for SSH to actually accept a login, not just answer the
   TCP port.
3. Syncs this repo to the pod (default: rsync the working tree, including
   uncommitted changes; excludes `data/`, `results/`, `archive*/`,
   `outputs/` by default -- pass `--sync-all` if the job needs those).
4. Runs the command on the pod, streaming output, inside the synced repo.
5. Pulls back any `--pull <path>` targets.
6. Terminates the pod -- but **only if this invocation created it**. A pod
   this script merely reused (found already running) is left alone unless
   `--kill-reused` was explicitly passed. This is deliberate: on this
   project, pods are sometimes left running on purpose for another
   concurrent session's job (see this repo's `CLAUDE.md` multi-machine
   section) -- auto-killing a pod you didn't create would be exactly the
   kind of cross-session interference that file warns against. Say so
   plainly if you're about to reuse a pod: which pod, and that it won't be
   killed automatically.

## When NOT to use this

- The job is small enough for local CPU/MPS -- don't spin up billed GPU time
  for something that runs in seconds locally.
- A pod is already up and mid-job for another purpose -- check what it's
  doing (`runpodctl pod list`) before assuming it's free to reuse.
