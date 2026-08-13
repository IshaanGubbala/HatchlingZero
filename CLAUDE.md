# Multi-machine dispatch relay

This repo is worked on from multiple machines: a Mac (primary dev
machine), a Raspberry Pi (`gubbi`, always-on relay hub), and a Windows
box with an RTX 3060 (real GPU compute for training/eval runs this
Mac's own hardware — MPS or CPU — is too slow or lacks CUDA for). If
you are a Claude Code session running in this repo on ANY of those
three machines, this file is for you: a real, working async job-dispatch
system already exists, already deployed, already tested — use it
instead of re-inventing SSH/scp coordination or assuming these machines
can't reach each other.

## The real topology

```text
Mac  <--Tailscale-->  Pi (hub)  <--Tailscale-->  Windows/RTX3060
                         |
                    Cloudflare Tunnel
                         |
                  public dashboard (phone/browser, any network)
```

The Pi is the dispatch hub. Windows polls the Pi's **outbox** for new
work; anyone (Mac, Pi, or a human on their phone via the dashboard) can
drop a request there and it gets picked up. Results come back into the
Pi's **inbox**.

**All three machines are already joined to the same Tailscale tailnet**
— no VPN/account setup needed, just use the Tailscale IPs below.
Tailscale IPs are stable across reboots/DHCP renewals; LAN IPs are NOT
(confirmed: the Pi's LAN IP changed after a routine reboot, its
Tailscale IP did not) — always prefer Tailscale IPs for anything you
write down or script against.

## Real, live components (as of 2026-08-13)

- **Relay server** on the Pi: `~/hz0a_transfer/serve.py`, HTTPS +
  self-signed cert, systemd service `hz0a-transfer.service` (enabled,
  auto-restarts, survives reboots), Tailscale-only, port 8899, token via
  the `X-Auth` header. Same server also runs on the Mac at
  `~/hz0a_transfer/serve.py` (own outbox/inbox, own token — a second,
  independent relay, not required for normal dispatch but still live).
  Real HTTP surface (use this directly — curl/requests, no SSH needed
  for any of it):
  - `GET  /`              list what's in this machine's outbox
  - `GET  /outbox/<name>` download a file from outbox
  - `GET  /inbox/<name>`  download a file from inbox
  - `PUT  /inbox/<name>`  upload a file (body = raw bytes) into inbox
  - `POST /chat/<name>`   append a plain-text chat message (body = raw
    UTF-8 text, `<name>` is your sender name — any name works, no
    allowlist)
  - `GET  /chat`          read the chat log, `"[idx] ts from: text"` per
    line; `?since=N` returns only messages after line index N (cheap
    incremental polling instead of re-fetching the whole history)
  There is still no `PUT /outbox/<name>` on this server (writing INTO
  someone's outbox goes through the dashboard's upload form instead,
  see below) — everything else is a plain HTTPS call.
- **Chat** is for short back-and-forth status/coordination text ("starting
  the run now", "done, check inbox") that doesn't need a full file —
  faster than the file-drop round trip for anything short. Real file
  transfers (checkpoints, request specs) still go through inbox/outbox.
  Chat messages persist in `~/hz0a_transfer/chat.jsonl` on the Pi (one
  JSON object per line: `{"ts", "from", "text"}`), shared between the
  relay server's own `/chat` routes and the dashboard's chat panel — post
  from either, read from either.
- **Status dashboard** on the Pi: `~/hz0a_transfer/dashboard.py`,
  systemd service `hz0a-dashboard.service`, bound to `127.0.0.1:8080`
  only. Exposed publicly via a Cloudflare quick tunnel
  (`hz0a-tunnel.service`) — the actual public URL changes on tunnel
  restart, check `~/hz0a_transfer/tunnel.log` on the Pi for the current
  one. Has a real file-upload form (routes to Pi inbox / Pi outbox /
  any configured peer's inbox via a dropdown), a live chat panel (post
  and read, same `chat.jsonl` the relay server uses), and shows live
  inbox/outbox contents for the Pi plus every peer configured in the
  `PEERS` dict at the top of `dashboard.py`. **Adding a new device**
  that runs its own `serve.py` instance (like the Mac does) needs no
  code restructuring — just one new entry in `PEERS` (label, Tailscale
  URL, token) and it gets its own status card and upload destination
  automatically. A device that doesn't run its own relay can still
  fully participate in chat and the Pi's inbox/outbox with nothing more
  than the Pi's Tailscale IP + token — `/chat/<name>`, `/inbox/<name>`,
  `/outbox/<name>` all accept any name, not a fixed machine list.
- **SSH**: key-based from the Mac (`ssh gubbipi`, shortcut already
  configured in `~/.ssh/config` on the Mac, points at the Pi's
  Tailscale IP). Reserve this for actual server maintenance (deploying
  new `serve.py`/`dashboard.py` code, restarting systemd services) —
  routine dispatch and result-checking should go through the HTTP API
  above, not SSH/scp. Real, observed failure mode: Tailscale SSH's own
  interception layer can require an interactive browser re-auth
  mid-session (`https://login.tailscale.com/a/...`), which blocks SSH
  entirely until a human clicks through — HTTP over the same Tailscale
  IP is unaffected by this, another reason to prefer it. From Windows
  or a fresh Mac session without the SSH shortcut, use the real
  credentials — see "Finding the real credentials" below.

## How to dispatch work (the pattern used all session)

1. Write a plain-text request file describing the job — what to run,
   why, what config, what to report back. Match the style of existing
   files in `~/hz0a_transfer/outbox/` on the Pi (there are many real
   examples there already — read a few before writing your first one).
2. Drop it in the Pi's `~/hz0a_transfer/outbox/` — via the dashboard's
   upload form (destination "Windows/RTX3060 (via Pi outbox)"), the
   only path that writes INTO outbox over HTTP; SSH/scp still works too
   but isn't the default anymore.
3. Poll for the reply. There's no inbox-listing endpoint (only outbox
   has `GET /` for listing) — either poll `GET /inbox/<expected-name>`
   for a name you agreed on in the request (404 until it lands), check
   the dashboard page (shows live inbox contents), or watch `/chat` for
   a status ping. A background Monitor polling loop is the pattern used
   this session for the SSH-based version — same idea works against the
   HTTP endpoints.
4. The other side (whichever machine picks up the job) reports back the
   same way: PUTs a result file into the Pi's inbox
   (`PUT /inbox/<name>`), and/or drops a quick heads-up on `/chat`.

This is real async, human-readable coordination — not a job queue API.
Treat every request/result file (and chat message) as a message to a
colleague on another machine: explain why, give exact commands, say
what to report back.

## Finding the real credentials (deliberately NOT in this file)

Auth tokens and the dashboard password are NOT committed to git (this
repo is public) — they live only in the actual deployed files on each
machine:

- Relay auth token (`X-Auth` header): read the `TOKEN` constant
  directly from `~/hz0a_transfer/serve.py` on whichever machine you're
  targeting (Pi and Mac each have their own, distinct token).
- Dashboard login: read `DASH_USER`/`DASH_PASS` from
  `~/hz0a_transfer/dashboard.py` on the Pi.
- Pi SSH access: ask the user directly if you don't already have it —
  don't guess or try common defaults.

If you're a fresh session and none of this is reachable yet (e.g. a
brand new Windows machine that's never touched this relay), ask the
user rather than assuming — this file describes what's real and
already working, not a spec to reimplement from scratch.

## Real, disclosed limitations

- The public dashboard URL is NOT stable (quick tunnel, no domain
  configured) — don't hardcode it anywhere durable, always re-check
  `tunnel.log` on the Pi.
- The relay servers are Tailscale-only by design, never port-forwarded
  — don't try to expose them more broadly without asking first.
- Windows/RTX3060 connectivity to the Pi has shown real, observed
  flakiness (LAN and Tailscale paths flipped between reachable/
  unreachable within minutes of each other on one occasion, self-
  recovered without intervention) — if dispatch seems stuck, that's a
  known real pattern to check, not necessarily a new problem.
