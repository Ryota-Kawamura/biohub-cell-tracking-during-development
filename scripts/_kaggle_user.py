"""Resolve the Kaggle username.

The CLI supports three ways to authenticate and each stores its data in a
different place, so the lookup lives in one spot:

    1. the KAGGLE_USERNAME environment variable
    2. ~/.kaggle/credentials.json  (`kaggle auth login`, OAuth)
    3. ~/.kaggle/kaggle.json       (legacy API key)

Only the `username` field is read, never the token.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

KAGGLE_DIR = pathlib.Path.home() / ".kaggle"

AUTH_HINT = """Kaggle authentication is not set up. Run one of:

  kaggle auth login                  (OAuth, opens a browser)

  or create a token at https://www.kaggle.com/settings/api and save it to
  ~/.kaggle/access_token (or set KAGGLE_API_TOKEN).
"""


def _from_json(path: pathlib.Path) -> str | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("username") or None
    except (json.JSONDecodeError, OSError):
        return None


def kaggle_username(override: str | None = None) -> str:
    """Return the username, or exit with a hint if none can be found."""
    for candidate in (
        override,
        os.environ.get("KAGGLE_USERNAME"),
        _from_json(KAGGLE_DIR / "credentials.json"),
        _from_json(KAGGLE_DIR / "kaggle.json"),
    ):
        if candidate:
            return candidate

    sys.exit(AUTH_HINT + "\nIf you are authenticated already, set KAGGLE_USERNAME.\n")
