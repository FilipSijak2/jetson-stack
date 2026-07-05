#!/usr/bin/env python3
"""Convenience wrapper: mark the newest matching anomaly event as FP."""

from __future__ import annotations

import sys

from mark_event import main


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--last",
        "--verdict",
        "FP",
        *sys.argv[1:],
    ]
    raise SystemExit(main())
