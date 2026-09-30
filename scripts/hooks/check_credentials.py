"""Block commits that contain Kaggle credentials.

Called from the pre-commit hook (`git config core.hooksPath scripts/hooks`).
Checks both file names and the added lines, and exits 1 on a match.
"""

from __future__ import annotations

import re
import subprocess
import sys

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BAD_NAMES = re.compile(r"(^|/)(kaggle\.json|credentials\.json|access_token)$|(^|/)\.kaggle/")
# Kaggle API tokens: the KGAT_ format and the legacy kaggle.json "key".
BAD_CONTENT = re.compile(r"KGAT_[A-Za-z0-9]{16,}|\"key\"\s*:\s*\"[0-9a-f]{32}\"")


def main() -> int:
    names = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    ).stdout.split()
    problems = [f"credential-like file: {n}" for n in names if BAD_NAMES.search(n)]

    diff = subprocess.run(
        ["git", "diff", "--cached", "-U0", "--diff-filter=ACMR"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    ).stdout
    current = ""
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
        elif line.startswith("+") and not line.startswith("+++") and BAD_CONTENT.search(line):
            problems.append(f"token-like string in: {current}")

    if problems:
        print("Commit blocked: possible credentials staged:", file=sys.stderr)
        for p in sorted(set(problems)):
            print(f"  {p}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
