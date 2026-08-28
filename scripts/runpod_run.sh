#!/usr/bin/env bash
# Run one command on a RunPod GPU pod, then auto-terminate the pod.
#
# Agent-agnostic on purpose: any agent that can run a shell command (Claude
# Code, Codex, a human) can call this directly. There is no CLI-specific
# wrapper logic here -- just runpodctl, rsync, and ssh.
#
# Reuse policy (deliberate, not the literal "always kill" reading of "auto
# kill the runpod"): if a pod is already running, this script reuses it but
# does NOT kill it on exit unless you pass --kill-reused. A pod this
# invocation creates itself IS always auto-terminated on exit (success,
# failure, or Ctrl-C) unless you pass --keep. This matters because pods are
# sometimes deliberately left running for another concurrent session's job
# -- see this repo's CLAUDE.md multi-machine section. Auto-killing a pod you
# didn't create is exactly the kind of cross-session interference that file
# warns against.
#
# Requires: runpodctl (authenticated), jq, rsync, ssh.
#
# Examples:
#   scripts/runpod_run.sh -- python scripts/some_cuda_benchmark.py --out results/x.json
#   scripts/runpod_run.sh --pull results/x.json -- python scripts/some_cuda_benchmark.py --out results/x.json
#   scripts/runpod_run.sh --gpu-id "NVIDIA A100 80GB PCIe" --ttl-minutes 60 -- nvidia-smi
#   scripts/runpod_run.sh --keep -- pip install -r requirements.txt   # leave pod up for a follow-up call
#   scripts/runpod_run.sh --sync none --keep -- python train.py       # reuse state from a prior --keep call

set -euo pipefail

GPU_ID="NVIDIA A40"
IMAGE="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
DISK_GB=20
TTL_MINUTES=180
NETWORK_VOLUME_ID=""
POD_NAME=""
SYNC_MODE="local"
REMOTE_DIR=""
SSH_KEY="$HOME/.ssh/id_ed25519"
KEEP=0
NO_REUSE=0
KILL_REUSED=0
WAIT_TIMEOUT="6m"
PULL_PATHS=()
EXTRA_EXCLUDES=()
NO_DEFAULT_EXCLUDES=0
COMMAND=()

usage() {
    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu-id) GPU_ID="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        --disk-gb) DISK_GB="$2"; shift 2 ;;
        --network-volume-id) NETWORK_VOLUME_ID="$2"; shift 2 ;;
        --ttl-minutes) TTL_MINUTES="$2"; shift 2 ;;
        --name) POD_NAME="$2"; shift 2 ;;
        --sync) SYNC_MODE="$2"; shift 2 ;;
        --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
        --ssh-key) SSH_KEY="$2"; shift 2 ;;
        --keep) KEEP=1; shift ;;
        --no-reuse) NO_REUSE=1; shift ;;
        --kill-reused) KILL_REUSED=1; shift ;;
        --wait-timeout) WAIT_TIMEOUT="$2"; shift 2 ;;
        --pull) PULL_PATHS+=("$2"); shift 2 ;;
        --rsync-exclude) EXTRA_EXCLUDES+=("$2"); shift 2 ;;
        --sync-all) NO_DEFAULT_EXCLUDES=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; COMMAND=("$@"); break ;;
        *) echo "unknown flag: $1 (did you forget '--' before the command?)" >&2; exit 2 ;;
    esac
done

if [[ ${#COMMAND[@]} -eq 0 ]]; then
    echo "no command given -- pass one after '--', e.g.: $0 -- python script.py" >&2
    exit 2
fi
if [[ "$SYNC_MODE" != "local" && "$SYNC_MODE" != "git" && "$SYNC_MODE" != "none" ]]; then
    echo "--sync must be local|git|none, got: $SYNC_MODE" >&2
    exit 2
fi
for bin in runpodctl jq rsync ssh; do
    command -v "$bin" >/dev/null 2>&1 || { echo "required tool not found: $bin" >&2; exit 2; }
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
REPO_NAME="$(basename "$REPO_ROOT")"
[[ -n "$REMOTE_DIR" ]] || REMOTE_DIR="/workspace/$REPO_NAME"
[[ -n "$POD_NAME" ]] || POD_NAME="agent-run-$(date -u +%Y%m%dT%H%M%SZ)"

POD_ID=""
POD_IP=""
POD_PORT=""
CREATED_BY_SCRIPT=0

log() { echo "[runpod_run] $*" >&2; }

find_running_pod() {
    runpodctl pod list -o json 2>/dev/null | jq -r '.[0].id // empty'
}

# Installed BEFORE any pod is created/looked up (real, observed bug
# 2026-08-27: a `runpodctl pod create --wait` timeout -- the pod DOES
# get created, it just isn't reachable yet -- tripped `set -e` on the
# CREATE_JSON assignment before POD_ID/CREATED_BY_SCRIPT were set and
# before this trap used to be installed further down, leaking a
# billing pod with no cleanup at all). Cleanup is a no-op while POD_ID
# is still empty, so installing it this early is always safe.
cleanup() {
    local exit_code=$?
    if [[ -z "$POD_ID" ]]; then
        exit "$exit_code"
    fi
    if [[ $CREATED_BY_SCRIPT -eq 1 && $KEEP -eq 0 ]]; then
        log "terminating pod $POD_ID (created by this run)"
        runpodctl pod remove "$POD_ID" >/dev/null 2>&1 || log "warning: failed to remove pod $POD_ID -- check runpodctl pod list"
    elif [[ $CREATED_BY_SCRIPT -eq 0 && $KILL_REUSED -eq 1 && $KEEP -eq 0 ]]; then
        log "terminating reused pod $POD_ID (--kill-reused was set)"
        runpodctl pod remove "$POD_ID" >/dev/null 2>&1 || log "warning: failed to remove pod $POD_ID -- check runpodctl pod list"
    elif [[ $KEEP -eq 1 ]]; then
        log "leaving pod $POD_ID running (--keep). Remember to remove it later: runpodctl pod remove $POD_ID"
    else
        log "leaving reused pod $POD_ID running (not created by this invocation, --kill-reused not set)"
    fi
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

if [[ $NO_REUSE -eq 0 ]]; then
    EXISTING_ID="$(find_running_pod || true)"
    if [[ -n "$EXISTING_ID" ]]; then
        POD_ID="$EXISTING_ID"
        GET_JSON="$(runpodctl pod get "$POD_ID" -o json)"
        POD_IP="$(jq -r '.ssh.ip' <<<"$GET_JSON")"
        POD_PORT="$(jq -r '.ssh.port' <<<"$GET_JSON")"
        POD_NAME_ACTUAL="$(jq -r '.name' <<<"$GET_JSON")"
        log "reusing already-running pod $POD_ID ('$POD_NAME_ACTUAL') at $POD_IP:$POD_PORT -- will NOT auto-kill it unless --kill-reused was passed"
    fi
fi

if [[ -z "$POD_ID" ]]; then
    TERMINATE_AT="$(date -u -v+"${TTL_MINUTES}"M +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d "+${TTL_MINUTES} minutes" +%Y-%m-%dT%H:%M:%SZ)"
    VOLUME_ARGS=()
    [[ -n "$NETWORK_VOLUME_ID" ]] && VOLUME_ARGS=(--network-volume-id "$NETWORK_VOLUME_ID")
    log "no running pod found, creating one: gpu='$GPU_ID' image=$IMAGE disk=${DISK_GB}GB ttl=${TTL_MINUTES}m network_volume='${NETWORK_VOLUME_ID:-none}' (auto-terminate at $TERMINATE_AT as a hard safety net)"
    set +e
    CREATE_JSON="$(runpodctl pod create \
        --image "$IMAGE" \
        --gpu-id "$GPU_ID" \
        --container-disk-in-gb "$DISK_GB" \
        --ports "22/tcp" \
        --name "$POD_NAME" \
        --terminate-after "$TERMINATE_AT" \
        "${VOLUME_ARGS[@]}" \
        --wait --wait-timeout "$WAIT_TIMEOUT")"
    CREATE_EXIT=$?
    set -e
    # Even on a --wait timeout the pod itself was really created (it's just
    # not reachable yet) -- pull the id out regardless of CREATE_EXIT so the
    # trap above can clean it up instead of leaking a billing pod.
    MAYBE_ID="$(jq -r '.id // empty' <<<"$CREATE_JSON" 2>/dev/null || true)"
    if [[ -n "$MAYBE_ID" ]]; then
        POD_ID="$MAYBE_ID"
        CREATED_BY_SCRIPT=1
    fi
    if [[ $CREATE_EXIT -ne 0 ]]; then
        log "pod create failed/timed out (exit $CREATE_EXIT): $CREATE_JSON"
        [[ -n "$POD_ID" ]] && log "pod $POD_ID was created but never became reachable -- it will be terminated on exit"
        exit 1
    fi
    POD_IP="$(jq -r '.ssh.ip' <<<"$CREATE_JSON")"
    POD_PORT="$(jq -r '.ssh.port' <<<"$CREATE_JSON")"
    COST_HR="$(jq -r '.costPerHr' <<<"$CREATE_JSON")"
    log "pod $POD_ID ready at $POD_IP:$POD_PORT (\$$COST_HR/hr)"
fi

SSH_OPTS=(-p "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=20)
SSH_TARGET="root@$POD_IP"

# Wait for the ssh daemon itself to accept a real login, not just the TCP
# port -- `pod create --wait` only confirms the TCP banner answers.
for i in $(seq 1 30); do
    if ssh "${SSH_OPTS[@]}" -o BatchMode=yes "$SSH_TARGET" true 2>/dev/null; then
        break
    fi
    [[ $i -eq 30 ]] && { log "ssh never became usable on $POD_ID"; exit 1; }
    sleep 2
done

ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "mkdir -p '$REMOTE_DIR'"

case "$SYNC_MODE" in
    local)
        RSYNC_EXCLUDES=(--exclude .git --exclude __pycache__ --exclude '*.pyc' --exclude .DS_Store --exclude target)
        if [[ $NO_DEFAULT_EXCLUDES -eq 0 ]]; then
            RSYNC_EXCLUDES+=(--exclude data --exclude results --exclude archive --exclude archive2 --exclude outputs)
        fi
        for pat in "${EXTRA_EXCLUDES[@]:-}"; do
            [[ -n "$pat" ]] && RSYNC_EXCLUDES+=(--exclude "$pat")
        done
        log "syncing working tree to $SSH_TARGET:$REMOTE_DIR (rsync, default excludes: data/ results/ archive*/ outputs/ target/ -- use --sync-all to disable)"
        # --delete is unsafe when a network volume is attached: multiple
        # concurrent pods mounting the SAME volume at the SAME REMOTE_DIR
        # (see --network-volume-id) share one directory tree, so one pod's
        # sync deleting files "not present locally" can wipe another
        # still-running pod's runtime logs/output out from under it (real,
        # observed 2026-08-24 -- training processes survived since they
        # already had the file open, but `tail`/`ls` on the log path broke
        # mid-run on 3 concurrent pods). Safe to keep --delete for the
        # normal ephemeral-disk case (each pod gets its own untouched tree).
        DELETE_FLAG=(--delete)
        [[ -n "$NETWORK_VOLUME_ID" ]] && DELETE_FLAG=()
        # -rlt (not -a): real, observed failure on RunPod containers --
        # -a implies -o/-g (preserve owner/group), which needs chown()
        # privileges the container doesn't have even as root (2026-08-27,
        # every file in the sync failing with "Operation not permitted"
        # and the whole rsync exiting nonzero, killing the sync before any
        # training ran). -rlt keeps recursion/symlinks/timestamps, drops
        # the owner/group preservation that was never needed here anyway
        # (single-user pods, no multi-owner file tree).
        rsync -rltz "${DELETE_FLAG[@]:-}" -e "ssh ${SSH_OPTS[*]}" "${RSYNC_EXCLUDES[@]}" "$REPO_ROOT/" "$SSH_TARGET:$REMOTE_DIR/"
        ;;
    git)
        ORIGIN_URL="$(git -C "$REPO_ROOT" remote get-url origin)"
        HEAD_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
        log "syncing via git clone/checkout of $HEAD_COMMIT (must be pushed to origin already)"
        ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "
            set -e
            if [ -d '$REMOTE_DIR/.git' ]; then
                cd '$REMOTE_DIR' && git fetch origin
            else
                git clone '$ORIGIN_URL' '$REMOTE_DIR'
                cd '$REMOTE_DIR'
            fi
            git checkout '$HEAD_COMMIT'
        "
        ;;
    none)
        log "skipping code sync (--sync none), assuming $REMOTE_DIR already has what it needs"
        ;;
esac

log "running: ${COMMAND[*]}"
# Real, observed bug 2026-08-27: a single long-lived foreground SSH session
# held open for a multi-hour command dropped silently mid-run (network
# blip / laptop sleep, exact cause unconfirmed) and killed the remote
# process with it, since it was attached directly to that session's pty
# with no nohup. Decoupled here: launch via nohup+disown on the remote
# side (same pattern this repo's own manual dispatches use throughout),
# then poll for a completion marker via SEPARATE short-lived SSH calls --
# one dropped poll just retries, it can't kill a job it was never attached
# to.
REMOTE_MARKER=".runpod_run_$(date +%s)_$$"
# Real bug, found 2026-08-27/28: ${COMMAND[*]} space-joins the array and
# loses any quoting a caller embedded in a single element (e.g. passing
# `bash -c 'cmd1 && cmd2'` as one COMMAND element) -- the remote shell then
# re-tokenizes that flattened string from scratch, so `bash -c cmd1 && cmd2`
# gets parsed as bash -c consuming only the single word right after -c as
# its command-string, with everything else becoming inert positional
# params. Concretely: the FIRST of two &&-chained commands silently
# no-ops (bash -c <word> with stdin=/dev/null exits ~instantly), and only
# the SECOND command actually runs -- discovered when a real two-arm n-gram
# dispatch only produced results for its second arm. printf %q re-quotes
# each array element so the remote bash -c reconstructs the exact same
# argument boundaries the caller passed in.
REMOTE_CMD="$(printf '%q ' "${COMMAND[@]}")"
ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "cd '$REMOTE_DIR' && nohup bash -c '$REMOTE_CMD; echo \$? > $REMOTE_MARKER.exit' > $REMOTE_MARKER.log 2>&1 < /dev/null & disown; sleep 1; true"
log "launched (marker $REMOTE_MARKER), polling for completion every 15s -- tolerant of transient SSH drops"
CMD_EXIT=""
while [[ -z "$CMD_EXIT" ]]; do
    sleep 15
    CMD_EXIT="$(ssh "${SSH_OPTS[@]}" -o ConnectTimeout=10 "$SSH_TARGET" "cat '$REMOTE_DIR/$REMOTE_MARKER.exit' 2>/dev/null" 2>/dev/null || true)"
done
log "remote command finished with exit code $CMD_EXIT"
ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "cat '$REMOTE_DIR/$REMOTE_MARKER.log'" >&2 || true

for path in "${PULL_PATHS[@]:-}"; do
    [[ -n "$path" ]] || continue
    log "pulling back $path"
    mkdir -p "$(dirname "$REPO_ROOT/$path")"
    rsync -az -e "ssh ${SSH_OPTS[*]}" "$SSH_TARGET:$REMOTE_DIR/$path" "$REPO_ROOT/$path" || log "warning: failed to pull back $path"
done

exit $CMD_EXIT
