#!/usr/bin/env python3
"""List or review historical anomaly events without modifying events.jsonl."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation_common import (
    append_jsonl,
    filter_records_by_date,
    latest_reviews,
    load_jsonl,
    normalize_date_filter,
)


VERDICTS = {"FP", "TP", "UNSURE"}


def default_artifact_root() -> Path:
    return Path(os.getenv("JETSON_ARTIFACT_ROOT", "./anomaly_logs"))


def event_summary(event: Dict[str, Any], verdict: str = "") -> str:
    localization = event.get("localization") or {}
    files = event.get("jetson_files") or {}
    return (
        f"{str(event.get('id') or '-'):12} "
        f"{str(event.get('timestamp') or '-'):26} "
        f"conf={float(event.get('confidence', 0.0)):.2f} "
        f"track={event.get('track_id') if event.get('track_id') is not None else '-'} "
        f"distance={float(localization.get('distance_m', 0.0)):.2f}m "
        f"review={verdict or '-':7} "
        f"image={files.get('annotated_image') or files.get('original_image') or '-'}"
    )


def find_event(
    events: List[Dict[str, Any]],
    event_id: Optional[str],
    use_last: bool,
) -> Optional[Dict[str, Any]]:
    if use_last:
        return events[-1] if events else None
    wanted = str(event_id or "").strip()
    return next(
        (event for event in events if str(event.get("id") or "") == wanted),
        None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List anomaly events or append an FP/TP/UNSURE review."
    )
    parser.add_argument("event_id", nargs="?", help="Event id, e.g. anom_00042")
    parser.add_argument(
        "--verdict",
        choices=sorted(VERDICTS),
        help="Review verdict to append",
    )
    parser.add_argument(
        "--last",
        action="store_true",
        help="Select the newest event after applying --date",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List matching events without changing reviews",
    )
    parser.add_argument(
        "--date",
        default="all",
        help="all, today, yesterday or YYYY-MM-DD",
    )
    parser.add_argument("--note", default="", help="Optional short review note")
    parser.add_argument(
        "--reviewer",
        default=os.getenv("USER", "operator"),
        help="Reviewer name stored with the verdict",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=default_artifact_root(),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_root = args.artifact_root.resolve()
    events_path = artifact_root / "events.jsonl"
    reviews_path = artifact_root / "event_reviews.jsonl"
    try:
        date_filter = normalize_date_filter(args.date)
    except ValueError:
        print(f"Invalid --date value: {args.date}", file=sys.stderr)
        return 2

    all_events = load_jsonl(events_path)
    events = filter_records_by_date(all_events, date_filter)
    reviews = latest_reviews(load_jsonl(reviews_path))
    if args.list:
        if not events:
            print(f"No events found in {events_path} for date={args.date}")
            return 0
        for event in events:
            event_id = str(event.get("id") or "")
            verdict = str((reviews.get(event_id) or {}).get("verdict") or "")
            print(event_summary(event, verdict))
        return 0

    if not args.verdict:
        print("Use --list or provide --verdict FP|TP|UNSURE", file=sys.stderr)
        return 2
    if not args.event_id and not args.last:
        print("Provide event_id or use --last", file=sys.stderr)
        return 2

    event = find_event(events, args.event_id, args.last)
    if event is None:
        print(
            f"Event not found in {events_path} for date={args.date}",
            file=sys.stderr,
        )
        return 1
    event_id = str(event.get("id") or "")
    review = {
        "event_id": event_id,
        "verdict": args.verdict,
        "timestamp": datetime.now().astimezone().isoformat(),
        "event_timestamp": event.get("timestamp"),
        "reviewer": args.reviewer,
        "note": args.note,
    }
    append_jsonl(reviews_path, review)
    print(f"Saved {args.verdict} review for {event_id} -> {reviews_path}")
    print(event_summary(event, args.verdict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
