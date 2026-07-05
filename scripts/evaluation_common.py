#!/usr/bin/env python3
"""Shared standard-library helpers for anomaly evaluation scripts."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return records
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"[WARN] {path}:{line_number}: invalid JSON: {exc}")
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def parse_timestamp(value: Any) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed.astimezone()


def normalize_date_filter(value: str) -> Optional[str]:
    normalized = str(value or "all").strip().lower()
    today = datetime.now().astimezone().date()
    if normalized in {"", "all"}:
        return None
    if normalized == "today":
        return today.isoformat()
    if normalized == "yesterday":
        return (today - timedelta(days=1)).isoformat()
    return datetime.strptime(normalized, "%Y-%m-%d").date().isoformat()


def filter_records_by_date(
    records: Iterable[Dict[str, Any]], date_filter: Optional[str]
) -> List[Dict[str, Any]]:
    if date_filter is None:
        return list(records)
    selected = []
    for record in records:
        timestamp = parse_timestamp(record.get("timestamp"))
        if timestamp is not None and timestamp.date().isoformat() == date_filter:
            selected.append(record)
    return selected


def latest_reviews(
    records: Iterable[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for record in records:
        event_id = str(record.get("event_id") or "").strip()
        if event_id:
            latest[event_id] = record
    return latest
