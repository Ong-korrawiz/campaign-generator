"""Persistence for generated concepts and per-generation feedback."""

from __future__ import annotations

import json
from threading import Lock
from typing import Any, Protocol


class FeedbackStore(Protocol):
    def save_generation(self, generation_id: str, value: dict[str, Any]) -> None: ...
    def submit(self, generation_id: str, ratings: dict[str, str]) -> dict[str, str] | None: ...


class MemoryFeedbackStore:
    """Process-local store for local development and API tests."""

    def __init__(self) -> None:
        self.generations: dict[str, dict[str, Any]] = {}
        self.feedback: dict[str, dict[str, str]] = {}
        self._lock = Lock()

    def save_generation(self, generation_id: str, value: dict[str, Any]) -> None:
        with self._lock:
            self.generations[generation_id] = value

    def submit(self, generation_id: str, ratings: dict[str, str]) -> dict[str, str] | None:
        with self._lock:
            if generation_id not in self.generations:
                return None
            current = self.feedback.setdefault(generation_id, {})
            current.update(ratings)
            return current.copy()


class GCSFeedbackStore:
    """Use one atomic GCS object per generation's latest ratings."""

    def __init__(self, bucket_name: str) -> None:
        from google.cloud import storage

        self.bucket = storage.Client().bucket(bucket_name)

    def save_generation(self, generation_id: str, value: dict[str, Any]) -> None:
        self.bucket.blob(f"feedback/generations/{generation_id}.json").upload_from_string(
            json.dumps(value, ensure_ascii=False, sort_keys=True),
            content_type="application/json",
            if_generation_match=0,
        )

    def submit(self, generation_id: str, ratings: dict[str, str]) -> dict[str, str] | None:
        from google.api_core.exceptions import NotFound, PreconditionFailed

        generation = self.bucket.blob(f"feedback/generations/{generation_id}.json")
        try:
            generation.reload()
        except NotFound:
            return None
        feedback = self.bucket.blob(f"feedback/votes/{generation_id}.json")
        for _ in range(8):
            try:
                feedback.reload()
                expected_generation = int(feedback.generation)
                previous = json.loads(
                    self.bucket.blob(feedback.name, generation=expected_generation).download_as_bytes()
                )
            except NotFound:
                previous = {"ratings": {}}
                expected_generation = 0
            latest = {**previous["ratings"], **ratings}
            if latest == previous["ratings"]:
                feedback.reload()
                if int(feedback.generation) == expected_generation:
                    return latest
                continue
            try:
                feedback.upload_from_string(
                    json.dumps({"generation_id": generation_id, "ratings": latest}, sort_keys=True),
                    content_type="application/json",
                    if_generation_match=expected_generation,
                )
                return latest
            except PreconditionFailed:
                continue
        raise RuntimeError("feedback_concurrent_update")
