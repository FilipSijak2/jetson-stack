import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MARK_EVENT = REPO_ROOT / "scripts" / "mark_event.py"
GENERATE_REPORT = REPO_ROOT / "scripts" / "generate_cv_report.py"


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


class EvaluationToolsTest(unittest.TestCase):
    def test_can_mark_yesterdays_last_event_as_false_positive(self):
        yesterday = datetime.now().astimezone() - timedelta(days=1)
        events = [
            {
                "id": "anom_00001",
                "timestamp": yesterday.replace(hour=10).isoformat(),
                "confidence": 0.6,
            },
            {
                "id": "anom_00002",
                "timestamp": yesterday.replace(hour=11).isoformat(),
                "confidence": 0.8,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "events.jsonl", events)
            result = subprocess.run(
                [
                    sys.executable,
                    str(MARK_EVENT),
                    "--artifact-root",
                    str(root),
                    "--last",
                    "--date",
                    "yesterday",
                    "--verdict",
                    "FP",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            reviews = [
                json.loads(line)
                for line in (root / "event_reviews.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(reviews[-1]["event_id"], "anom_00002")
            self.assertEqual(reviews[-1]["verdict"], "FP")

    def test_report_uses_reviews_and_automatic_performance_metrics(self):
        now = datetime.now().astimezone().isoformat()
        events = [
            {
                "id": "anom_00001",
                "timestamp": now,
                "confidence": 0.9,
                "track_id": 1,
                "segmentation_mask_used": True,
                "localization": {
                    "distance_m": 1.0,
                    "distance_source": "depth",
                    "distance_uncertainty_m": 0.05,
                },
            },
            {
                "id": "anom_00002",
                "timestamp": now,
                "confidence": 0.6,
                "track_id": 2,
                "segmentation_mask_used": True,
                "localization": {
                    "distance_m": 1.5,
                    "distance_source": "laser",
                    "distance_uncertainty_m": 0.1,
                },
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "events.jsonl", events)
            write_jsonl(
                root / "event_reviews.jsonl",
                [{"event_id": "anom_00002", "verdict": "FP", "timestamp": now}],
            )
            write_jsonl(
                root / "inspections.jsonl",
                [{"request_id": "i1", "timestamp": now, "success": True}],
            )
            write_jsonl(
                root / "evaluation" / "performance.jsonl",
                [{"timestamp": now, "inference_ms": 20.0, "camera_fps": 15.0}],
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(GENERATE_REPORT),
                    "--artifact-root",
                    str(root),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(
                (root / "evaluation" / "latest" / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["events"], 2)
            self.assertEqual(summary["reviewed_false_positive"], 1)
            self.assertAlmostEqual(summary["precision_estimate"], 0.5)
            self.assertEqual(summary["inference_ms_mean"], 20.0)
            self.assertTrue(
                (root / "evaluation" / "latest" / "report.html").exists()
            )


if __name__ == "__main__":
    unittest.main()
