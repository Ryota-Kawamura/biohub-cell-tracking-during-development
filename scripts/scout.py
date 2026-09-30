"""List public notebooks for the competition and flag the ones not seen before.

Scores in notebook titles are unreliable (one titled "0.951 SOTA" actually
scored 0.944), so new notebooks are only flagged here; the real Public Score
is read from each notebook page.

    python scripts/scout.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")

COMP = "biohub-cell-tracking-during-development"
SEEN = pathlib.Path(__file__).resolve().parent.parent / "experiments" / "configs" / "seen_kernels.json"


def listing(sort: str, n: int) -> list[str]:
    out = subprocess.run(
        ["kaggle", "kernels", "list", "--competition", COMP,
         "--sort-by", sort, "--page-size", str(n)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    ).stdout
    refs = []
    for line in out.splitlines()[2:]:
        ref = line.split()[0] if line.split() else ""
        if "/" in ref:
            refs.append(ref.lstrip("b'"))
    return refs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    seen = json.loads(SEEN.read_text(encoding="utf-8")) if SEEN.exists() else {}
    found = {}
    for sort in ("dateRun", "voteCount", "scoreDescending"):
        for ref in listing(sort, args.n):
            found.setdefault(ref, sort)

    fresh = [r for r in found if r not in seen]
    print(f"{len(found)} public notebooks, {len(fresh)} new")
    for ref in fresh:
        print(f"  NEW  https://www.kaggle.com/code/{ref}")

    seen.update({r: found[r] for r in fresh})
    SEEN.parent.mkdir(parents=True, exist_ok=True)
    SEEN.write_text(json.dumps(seen, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nupdated {SEEN.name} ({len(seen)} known)")
    print("check the real Public Score on each page; titles are not reliable")


if __name__ == "__main__":
    main()
