"""One tick of the improvement loop.

It makes no decisions about what to try next. It gathers the state those
decisions need, records new scores, and does the one mechanical step:
submitting kernels that have finished.

    python scripts/cycle.py            # collect state, submit finished kernels
    python scripts/cycle.py --dry-run  # collect state only

Guard rails:
  - at most five submissions per UTC day, never the same slug twice
  - no new submissions after SUBMIT_CUTOFF (scoring takes up to ~19 h)
  - warn when the GPU estimate falls below the floor
  - never touches the final selection
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import subprocess
import sys

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "experiments" / "configs"
RESULTS = ROOT / "experiments" / "results" / "lb_results.json"
LEDGER = CONFIGS / "gpu_ledger.json"

COMP = "biohub-cell-tracking-during-development"
DAILY_LIMIT = 5
# Scoring takes up to ~19 h and the final deadline is 09-29 23:59 UTC.
SUBMIT_CUTOFF = dt.datetime(2026, 9, 29, 5, 0, tzinfo=dt.timezone.utc)
GPU_FLOOR_H = 2.0
# Weekly quota reset, read from the Kaggle account menu (Saturday ~00:13 UTC).
# It lands before the submission cutoff, so no reserve is needed until then.
GPU_RESET = dt.datetime(2026, 10, 3, 0, 13, tzinfo=dt.timezone.utc)
GPU_FLOOR_BEFORE_RESET_H = 0.5
LIGHT_RUN_H = 0.45

sys.path.insert(0, str(ROOT / "scripts"))
from _kaggle_user import kaggle_username  # noqa: E402


def kernel_status(user: str, slug: str) -> str:
    out = subprocess.run(
        ["kaggle", "kernels", "status", f"{user}/{slug}"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    ).stdout
    for token in ("COMPLETE", "RUNNING", "QUEUED", "ERROR", "CANCEL"):
        if token in out:
            return token
    return "UNKNOWN"


def gpu_left() -> float:
    d = json.loads(LEDGER.read_text(encoding="utf-8"))
    return float(d["weekly_hours"]) - sum(float(r["hours"]) for r in d["runs"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    user = kaggle_username()
    now = dt.datetime.now(dt.timezone.utc)

    # --- collect submissions and scores ----------------------------------------
    subs = []
    # The API pages at 20 by default; with more submissions the older ones would
    # look unsubmitted and get submitted again.
    for s in api.competition_submissions(COMP, page_size=200):
        d = s.to_dict() if hasattr(s, "to_dict") else {}
        raw = str(d.get("date") or "")
        try:
            when = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if when.tzinfo is None:
                when = when.replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
        desc = str(d.get("description") or "")
        subs.append({
            "slug": desc.split(":")[0].strip(),
            "when": when,
            # A notebook timeout comes back as COMPLETE with an errorDescription.
            "status": "ERROR" if d.get("errorDescription") else str(d.get("status") or "").split(".")[-1],
            "score": d.get("publicScore"),
            "desc": desc,
        })

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    known = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    fresh = []
    for s in subs:
        if s["score"] in (None, ""):
            continue
        if s["slug"] not in known:
            known[s["slug"]] = {"score": float(s["score"]), "desc": s["desc"],
                                "scored_seen_at": now.isoformat(timespec="minutes")}
            fresh.append(s)
    RESULTS.write_text(json.dumps(known, indent=1, ensure_ascii=False, sort_keys=True) + "\n",
                       encoding="utf-8")

    submitted = {s["slug"] for s in subs if "ERROR" not in s["status"]}
    submitted |= set(known)  # anything already scored counts as submitted, whatever the API returns
    used_today = sum(1 for s in subs if s["when"].date() == now.date())
    slots = max(0, DAILY_LIMIT - used_today)
    pending = [s for s in subs if s["score"] in (None, "") and "ERROR" not in s["status"]]

    # --- submit finished, not yet submitted kernels ---------------------------
    queue = []
    for cfg in sorted(CONFIGS.glob("batch*.json")):
        specs = json.loads(cfg.read_text(encoding="utf-8"))
        for slug, spec in specs.items():
            if slug in submitted:
                continue
            queue.append((cfg, slug, spec))

    actions = []
    for cfg, slug, spec in queue:
        st = kernel_status(user, slug)
        if st != "COMPLETE":
            actions.append(f"  wait    {slug:26s} kernel={st}")
            continue
        if now >= SUBMIT_CUTOFF:
            actions.append(f"  stop    {slug:26s} past the submission cutoff {SUBMIT_CUTOFF:%m-%d %H:%M} UTC")
            continue
        if slots <= 0:
            actions.append(f"  no slot {slug:26s} today's submissions are used up")
            continue
        if args.dry_run:
            actions.append(f"  (dry)   {slug:26s} would submit")
            continue
        try:
            api.competition_submit_code(
                file_name="submission.csv",
                message=f"{slug}: {spec.get('note', '')}",
                competition=COMP,
                kernel=f"{user}/{slug}",
                kernel_version=int(spec.get("version", 1)),
            )
            slots -= 1
            actions.append(f"  submit  {slug:26s} v{spec.get('version', 1)}")
        except Exception as exc:
            resp = getattr(exc, "response", None)
            msg = resp.json().get("error", {}).get("message") if resp is not None else str(exc)
            actions.append(f"  failed  {slug:26s} {msg}")

    # --- report ------------------------------------------------------------------
    reset = (now + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    left = gpu_left()
    floor = GPU_FLOOR_BEFORE_RESET_H if now < GPU_RESET else GPU_FLOOR_H
    print(f"now {now:%Y-%m-%d %H:%M} UTC")
    print(f"submissions today {DAILY_LIMIT - slots}/{DAILY_LIMIT}, {slots} left"
          f" | day resets in {(reset - now).total_seconds() / 3600:.1f} h"
          f" | cutoff in {(SUBMIT_CUTOFF - now).total_seconds() / 3600:.1f} h")
    print(f"GPU estimate {left:.2f} h left (floor {floor} h, {int(max(0, left - floor) / LIGHT_RUN_H)} light runs)"
          f" | weekly reset in {(GPU_RESET - now).total_seconds() / 3600:.1f} h ({GPU_RESET:%m-%d %H:%M} UTC)")
    if left < floor:
        print("  ! below the GPU floor: do not launch new experiments")
    print()
    print("newly scored:" if fresh else "newly scored: none")
    for s in fresh:
        print(f"  {float(s['score']):.3f}  {s['slug']}")
    print(f"pending: {len(pending)}  {[p['slug'] for p in pending]}")
    print()
    print("submission queue:" if actions else "submission queue: empty")
    for a in actions:
        print(a)
    print()
    # Scheduled non-submission runs (experiments/configs/pending_runs.json).
    pending_runs = CONFIGS / "pending_runs.json"
    if pending_runs.exists():
        for slug, job in json.loads(pending_runs.read_text(encoding="utf-8")).items():
            if job.get("done"):
                continue
            due = dt.datetime.fromisoformat(job["after"].replace("Z", "+00:00"))
            state = "DUE" if now >= due else f"after {due:%m-%d %H:%M} UTC"
            what = job.get("action") or f"python scripts/push_kernel.py -p {job['path']} --run"
            print(f"scheduled: {slug} [{state}]  {what}")
        print()
    best = sorted(known.items(), key=lambda kv: -kv[1]["score"])[:5]
    print("best 5:")
    for slug, r in best:
        print(f"  {r['score']:.3f}  {slug}")


if __name__ == "__main__":
    main()
