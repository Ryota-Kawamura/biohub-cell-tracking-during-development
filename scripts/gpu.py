"""Estimate the remaining weekly GPU quota.

Kaggle does not expose the quota through its API, so every run is booked in
experiments/configs/gpu_ledger.json. The number shown in the Kaggle UI is the
source of truth: when the two drift apart, update the ledger's calibration.
"""

from __future__ import annotations

import json
import pathlib
import sys

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")

LEDGER = pathlib.Path(__file__).resolve().parent.parent / "experiments" / "configs" / "gpu_ledger.json"
LIGHT_RUN_H = 0.45


def main() -> None:
    d = json.loads(LEDGER.read_text(encoding="utf-8"))
    total = sum(float(r["hours"]) for r in d["runs"])
    cap = float(d["weekly_hours"])
    cal = d["calibration"]

    print(f"weekly quota     {cap:.0f} h")
    print(f"booked runs      {total:.2f} h  ({len(d['runs'])} runs)")
    print(f"estimated left   {cap - total:.2f} h")
    if cal.get("ui_available_hours") is not None:
        drift = (cap - total) - float(cal["ui_available_hours"])
        print(f"UI at calibration {float(cal['ui_available_hours']):.2f} h left "
              f"(estimate drift {drift:+.2f} h)")
    print(f"calibrated       {cal['at']}")
    if "reset" in d:
        print(f"next reset       {d['reset']['next_at']}")
    print()
    print(f"about {int((cap - total) / LIGHT_RUN_H)} light submission runs ({LIGHT_RUN_H} h each) left")
    print()
    print("heaviest runs:")
    for r in sorted(d["runs"], key=lambda r: -float(r["hours"]))[:4]:
        print(f"  {float(r['hours']):5.2f} h  {r['kernel']} v{r['version']}  {r.get('note', '')}")


if __name__ == "__main__":
    main()
