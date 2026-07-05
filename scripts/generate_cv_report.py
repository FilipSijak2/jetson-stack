#!/usr/bin/env python3
"""Generate a self-contained HTML/JSON report from automatic anomaly logs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from evaluation_common import (
    filter_records_by_date,
    latest_reviews,
    load_jsonl,
    normalize_date_filter,
)


def default_artifact_root() -> Path:
    return Path(os.getenv("JETSON_ARTIFACT_ROOT", "./anomaly_logs"))


def finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [value for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def percentile(values: Iterable[Optional[float]], fraction: float) -> Optional[float]:
    clean = sorted(value for value in values if value is not None)
    if not clean:
        return None
    index = max(0, min(len(clean) - 1, round((len(clean) - 1) * fraction)))
    return clean[index]


def format_number(value: Optional[float], digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def event_verdict(
    event: Dict[str, Any], reviews: Dict[str, Dict[str, Any]]
) -> str:
    review = reviews.get(str(event.get("id") or "")) or {}
    return str(review.get("verdict") or "ASSUMED_TP").upper()


def threshold_rows(
    events: List[Dict[str, Any]],
    reviews: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    assumed_positive_total = sum(
        event_verdict(event, reviews) not in {"FP", "UNSURE"}
        for event in events
    )
    rows = []
    for integer in range(20, 91, 5):
        threshold = integer / 100.0
        selected = [
            event
            for event in events
            if float(event.get("confidence", 0.0)) >= threshold
        ]
        fp = sum(event_verdict(event, reviews) == "FP" for event in selected)
        tp = sum(
            event_verdict(event, reviews) not in {"FP", "UNSURE"}
            for event in selected
        )
        precision = tp / (tp + fp) if tp + fp else None
        recall_proxy = (
            tp / assumed_positive_total if assumed_positive_total else None
        )
        f1_proxy = (
            2.0 * precision * recall_proxy / (precision + recall_proxy)
            if precision is not None
            and recall_proxy is not None
            and precision + recall_proxy > 0.0
            else None
        )
        rows.append(
            {
                "threshold": threshold,
                "events": len(selected),
                "assumed_tp": tp,
                "reviewed_fp": fp,
                "precision_estimate": precision,
                "recall_proxy": recall_proxy,
                "f1_proxy": f1_proxy,
            }
        )
    return rows


def relative_artifact_link(path_value: Any, artifact_root: Path, output: Path) -> str:
    raw = str(path_value or "").strip()
    if not raw:
        return ""
    path = Path(raw)
    try:
        if path.is_absolute():
            try:
                relative = path.relative_to(artifact_root)
            except ValueError:
                parts = path.parts
                anomaly_index = next(
                    (
                        index
                        for index, part in enumerate(parts)
                        if part == "anomaly_logs"
                    ),
                    None,
                )
                if anomaly_index is None:
                    return ""
                relative = Path(*parts[anomaly_index + 1 :])
        else:
            relative = path
        target = artifact_root / relative
        return os.path.relpath(target, output).replace("\\", "/")
    except (ValueError, OSError):
        return ""


def write_events_csv(
    path: Path,
    events: List[Dict[str, Any]],
    reviews: Dict[str, Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "id",
                "timestamp",
                "verdict",
                "confidence",
                "track_id",
                "segmentation_mask_used",
                "distance_m",
                "distance_source",
                "uncertainty_m",
                "object_x",
                "object_y",
            ]
        )
        for event in events:
            localization = event.get("localization") or {}
            pose = event.get("object_pose_map") or {}
            writer.writerow(
                [
                    event.get("id"),
                    event.get("timestamp"),
                    event_verdict(event, reviews),
                    event.get("confidence"),
                    event.get("track_id"),
                    event.get("segmentation_mask_used"),
                    localization.get("distance_m"),
                    localization.get("distance_source"),
                    localization.get("distance_uncertainty_m"),
                    pose.get("x"),
                    pose.get("y"),
                ]
            )


def html_table(headers: List[str], rows: Iterable[Iterable[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(item))}</th>" for item in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(item))}</td>" for item in row)
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build_report(
    artifact_root: Path,
    output: Path,
    events: List[Dict[str, Any]],
    reviews: Dict[str, Dict[str, Any]],
    inspections: List[Dict[str, Any]],
    performance: List[Dict[str, Any]],
    date_label: str,
) -> Dict[str, Any]:
    verdict_counts = Counter(event_verdict(event, reviews) for event in events)
    fp_count = verdict_counts["FP"]
    unsure_count = verdict_counts["UNSURE"]
    assumed_tp = len(events) - fp_count - unsure_count
    precision_estimate = (
        assumed_tp / (assumed_tp + fp_count)
        if assumed_tp + fp_count
        else None
    )
    reviewed_count = sum(
        verdict_counts[verdict] for verdict in ("FP", "TP", "UNSURE")
    )
    localization_sources = Counter(
        str((event.get("localization") or {}).get("distance_source") or "unknown")
        for event in events
    )
    uncertainties = [
        finite_float(
            (event.get("localization") or {}).get("distance_uncertainty_m")
        )
        for event in events
    ]
    thresholds = threshold_rows(events, reviews)
    best_threshold = (
        max(
            thresholds,
            key=lambda row: (
                row["f1_proxy"] if row["f1_proxy"] is not None else -1.0
            ),
        )
        if events
        else None
    )
    inspection_success = sum(bool(item.get("success")) for item in inspections)
    inference_ms = [finite_float(item.get("inference_ms")) for item in performance]
    camera_fps = [finite_float(item.get("camera_fps")) for item in performance]
    summary = {
        "date_filter": date_label,
        "events": len(events),
        "assumed_true_positive": assumed_tp,
        "reviewed_false_positive": fp_count,
        "unsure": unsure_count,
        "reviewed_events": reviewed_count,
        "precision_estimate": precision_estimate,
        "tracking_coverage": (
            sum(event.get("track_id") is not None for event in events) / len(events)
            if events
            else None
        ),
        "segmentation_coverage": (
            sum(bool(event.get("segmentation_mask_used")) for event in events)
            / len(events)
            if events
            else None
        ),
        "mean_distance_uncertainty_m": mean(uncertainties),
        "p95_distance_uncertainty_m": percentile(uncertainties, 0.95),
        "localization_sources": dict(localization_sources),
        "inspection_attempts": len(inspections),
        "inspection_success": inspection_success,
        "inspection_success_rate": (
            inspection_success / len(inspections) if inspections else None
        ),
        "inference_ms_mean": mean(inference_ms),
        "inference_ms_p95": percentile(inference_ms, 0.95),
        "camera_fps_mean": mean(camera_fps),
        "recommended_confidence_threshold_proxy": (
            best_threshold["threshold"] if best_threshold else None
        ),
        "limitations": (
            "Recall, true F1, confusion matrix, absolute localization error and "
            "ground-truth segmentation IoU are unavailable without annotations. "
            "Unreviewed events are treated as assumed true positives."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_events_csv(output / "events.csv", events, reviews)

    fp_rows = []
    for event in events:
        if event_verdict(event, reviews) != "FP":
            continue
        files = event.get("jetson_files") or {}
        link = relative_artifact_link(
            files.get("annotated_image") or files.get("original_image"),
            artifact_root,
            output,
        )
        image_cell = f'<a href="{html.escape(link)}">image</a>' if link else "-"
        fp_rows.append(
            "<tr>"
            f"<td>{html.escape(str(event.get('id') or '-'))}</td>"
            f"<td>{html.escape(str(event.get('timestamp') or '-'))}</td>"
            f"<td>{float(event.get('confidence', 0.0)):.2f}</td>"
            f"<td>{image_cell}</td>"
            "</tr>"
        )
    threshold_table = html_table(
        ["threshold", "events", "assumed TP", "reviewed FP", "precision", "recall proxy", "F1 proxy"],
        [
            [
                f"{row['threshold']:.2f}",
                row["events"],
                row["assumed_tp"],
                row["reviewed_fp"],
                format_number(row["precision_estimate"], 3),
                format_number(row["recall_proxy"], 3),
                format_number(row["f1_proxy"], 3),
            ]
            for row in thresholds
        ],
    )
    source_table = html_table(
        ["distance source", "events"],
        sorted(localization_sources.items()),
    )
    cards = [
        ("Events", len(events)),
        ("Reviewed FP", fp_count),
        ("Precision estimate", format_number(precision_estimate, 3)),
        ("Inspection success", f"{inspection_success}/{len(inspections)}"),
        ("Inference mean", f"{format_number(mean(inference_ms))} ms"),
        ("Camera FPS", format_number(mean(camera_fps))),
    ]
    card_html = "".join(
        f'<div class="card"><span>{html.escape(str(label))}</span>'
        f"<strong>{html.escape(str(value))}</strong></div>"
        for label, value in cards
    )
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Jetson CV evaluation</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;background:#111827;color:#e5e7eb}}
h1,h2{{color:#f9fafb}} .cards{{display:flex;flex-wrap:wrap;gap:1rem}}
.card{{background:#1f2937;padding:1rem;border-radius:.6rem;min-width:10rem}}
.card span{{display:block;color:#9ca3af}} .card strong{{font-size:1.6rem}}
table{{border-collapse:collapse;width:100%;background:#1f2937;margin:1rem 0}}
th,td{{padding:.55rem;border:1px solid #374151;text-align:left}}
th{{background:#374151}} a{{color:#60a5fa}} .warning{{color:#fbbf24}}
</style></head><body>
<h1>Jetson CV evaluation</h1>
<p>Filter: {html.escape(date_label)}</p>
<div class="cards">{card_html}</div>
<p class="warning">{html.escape(summary["limitations"])}</p>
<h2>Localization sources</h2>{source_table}
<h2>Confidence threshold analysis</h2>
<p>Recommended proxy threshold: {format_number(summary["recommended_confidence_threshold_proxy"])}</p>
{threshold_table}
<h2>Reviewed false positives</h2>
<table><thead><tr><th>event</th><th>time</th><th>confidence</th><th>evidence</th></tr></thead>
<tbody>{''.join(fp_rows) or '<tr><td colspan="4">No reviewed FP events.</td></tr>'}</tbody></table>
<p>Machine-readable files: <a href="summary.json">summary.json</a>,
<a href="events.csv">events.csv</a>.</p>
</body></html>"""
    (output / "report.html").write_text(report, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate automatic CV evaluation report from Jetson logs."
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=default_artifact_root(),
    )
    parser.add_argument(
        "--date",
        default="all",
        help="all, today, yesterday or YYYY-MM-DD",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Default: <artifact-root>/evaluation/latest",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_root = args.artifact_root.resolve()
    output = (
        args.output.resolve()
        if args.output
        else artifact_root / "evaluation" / "latest"
    )
    try:
        date_filter = normalize_date_filter(args.date)
    except ValueError:
        print(f"Invalid --date value: {args.date}")
        return 2
    events = filter_records_by_date(
        load_jsonl(artifact_root / "events.jsonl"), date_filter
    )
    reviews = latest_reviews(load_jsonl(artifact_root / "event_reviews.jsonl"))
    inspections = filter_records_by_date(
        load_jsonl(artifact_root / "inspections.jsonl"), date_filter
    )
    performance = filter_records_by_date(
        load_jsonl(artifact_root / "evaluation" / "performance.jsonl"),
        date_filter,
    )
    summary = build_report(
        artifact_root,
        output,
        events,
        reviews,
        inspections,
        performance,
        date_filter or "all",
    )
    print(f"Report: {output / 'report.html'}")
    print(
        f"Events={summary['events']} FP={summary['reviewed_false_positive']} "
        f"precision_estimate={format_number(summary['precision_estimate'], 3)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
