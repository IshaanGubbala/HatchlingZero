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

## Real, live components (as of 2026-08-12)

- **Relay server** on the Pi: `~/hz0a_transfer/serve.py`, HTTPS +
  self-signed cert, systemd service `hz0a-transfer.service` (enabled,
  auto-restarts, survives reboots). Same server also runs on the Mac at
  `~/hz0a_transfer/serve.py` (own outbox/inbox, own token — a second,
  independent relay, not required for normal dispatch but still live).
- **Status dashboard** on the Pi: `~/hz0a_transfer/dashboard.py`,
  systemd service `hz0a-dashboard.service`, bound to `127.0.0.1:8080`
  only. Exposed publicly via a Cloudflare quick tunnel
  (`hz0a-tunnel.service`) — the actual public URL changes on tunnel
  restart, check `~/hz0a_transfer/tunnel.log` on the Pi for the current
  one. Has a real file-upload form (routes to Pi inbox / Pi outbox /
  Mac inbox via a dropdown) and shows live inbox/outbox contents for
  both the Pi and (over Tailscale) the Mac.
- **SSH**: key-based from the Mac (`ssh gubbipi`, shortcut already
  configured in `~/.ssh/config` on the Mac, points at the Pi's
  Tailscale IP). From Windows or a fresh Mac session without that
  shortcut, use the real credentials — see "Finding the real
  credentials" below.

## How to dispatch work (the pattern used all session)

1. Write a plain-text request file describing the job — what to run,
   why, what config, what to report back. Match the style of existing
   files in `~/hz0a_transfer/outbox/` on the Pi (there are many real
   examples there already — read a few before writing your first one).
2. Drop it in the Pi's `~/hz0a_transfer/outbox/` (via SSH/scp, or via
   the dashboard's upload form, destination "Windows/RTX3060 (via Pi
   outbox)").
3. Poll (or just periodically SSH and `ls`) the Pi's
   `~/hz0a_transfer/inbox/` for the reply. A background Monitor polling
   loop is the pattern used this session — see any recent conversation
   for the exact shell loop, or just check manually if you don't need
   to keep working in parallel.
4. The other side (whichever machine picks up the job) reports back the
   same way: writes a result file into the Pi's inbox.

This is real async, human-readable coordination — not a job queue API.
Treat every request/result file as a message to a colleague on another
machine: explain why, give exact commands, say what to report back.

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
