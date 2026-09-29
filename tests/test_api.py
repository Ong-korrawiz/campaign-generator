"""API contract tests with a fake model backend and no cloud credentials."""

from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from campaign_generator.api.app import create_app
from campaign_generator.api.backend import VLLMBackend
from campaign_generator.api.feedback import MemoryFeedbackStore
from campaign_generator.api.service import CampaignGenerator
from campaign_generator.schemas import CampaignGenerationResponse


class FakeBackend:
    def __init__(self, *, duplicate_once: bool = False) -> None:
        self.calls = []
        self.duplicate_once = duplicate_once
        self.duplicate_sent = False

    async def complete(self, *, messages, max_tokens):
        request = json.loads(messages[1]["content"])
        self.calls.append((request, max_tokens))
        name = f"{request['creative_angle'].title()} concept"
        if self.duplicate_once and request["creative_angle"] == "product proof" and not self.duplicate_sent:
            name = "Audience insight concept"
            self.duplicate_sent = True
        return json.dumps(
            {
                "campaign_direction": name,
                "campaign_description": "Show the supplied product proof in a clear example. Keep the execution consistent across the requested channels.",
                "audience_insight": "Hypothesis: the audience may value clear product information.",
                "key_message": "See how the supplied feature works.",
                "channel_plan": [
                    {
                        "channel": channel,
                        "role": "Explain the concept",
                        "format_note": "Use supplied proof only.",
                    }
                    for channel in request["channels"]
                ],
                "asset_plan": [
                    {
                        "priority": index + 1,
                        "asset": f"Campaign asset for {channel}",
                        "channel": channel,
                        "reason": "Supports the brief objective.",
                    }
                    for index, channel in enumerate(request["channels"])
                ],
            }
        )


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend(duplicate_once=True)
        self.feedback_store = MemoryFeedbackStore()
        self.client = TestClient(create_app(CampaignGenerator(self.backend), self.feedback_store))

    def tearDown(self):
        self.client.close()

    def brief(self):
        return {
            "industry": "productivity software",
            "brand": "Northstar Notes",
            "product": "meeting notes app",
            "target_audience": "team leads",
            "objective": "encourage free-trial signups",
            "channels": ["LinkedIn", "Email"],
            "proof_point": "The app turns meeting notes into assigned action items.",
            "constraints": ["Do not claim guaranteed time savings."],
        }

    def test_health_and_demo_do_not_call_backend(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        self.assertIn("Campaign concept demo", self.client.get("/").text)
        self.assertEqual(self.backend.calls, [])

    def test_generate_returns_three_concepts_and_retries_duplicate(self):
        response = self.client.post("/generate", json=self.brief())
        self.assertEqual(response.status_code, 200, response.text)
        concepts = response.json()["concepts"]
        CampaignGenerationResponse.model_validate(response.json())
        self.assertEqual(len(concepts), 3)
        self.assertEqual(len({item["campaign_direction"].casefold() for item in concepts}), 3)
        self.assertEqual(len(self.backend.calls), 4)
        self.assertTrue(all(item["campaign_description"] for item in concepts))
        self.assertTrue(all(item["proposed_kpis"] for item in concepts))
        self.assertTrue(all(item["budget_allocation"] == [] for item in concepts))
        self.assertTrue(all(max_tokens == 1536 for _, max_tokens in self.backend.calls))

    def test_budget_allocation_is_within_range_and_optional(self):
        brief = {**self.brief(), "budget_min": 100, "budget_max": 500, "currency": "THB"}
        response = self.client.post("/generate", json=brief)
        self.assertEqual(response.status_code, 200, response.text)
        for concept in response.json()["concepts"]:
            self.assertAlmostEqual(sum(item["amount"] for item in concept["budget_allocation"]), 300)
        partial = self.client.post("/generate", json={**self.brief(), "budget_min": 1})
        self.assertEqual(partial.status_code, 422)

    def test_single_sentence_description_gets_channel_execution_sentence(self):
        class OneSentenceBackend(FakeBackend):
            async def complete(self, *, messages, max_tokens):
                result = json.loads(await super().complete(messages=messages, max_tokens=max_tokens))
                result["campaign_description"] = "Show the supplied proof in a focused creative idea."
                return json.dumps(result)

        client = TestClient(create_app(CampaignGenerator(OneSentenceBackend())))
        try:
            response = client.post("/generate", json=self.brief())
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(
                all("LinkedIn, Email" in item["campaign_description"] for item in response.json()["concepts"])
            )
        finally:
            client.close()

    def test_body_size_and_schema_errors(self):
        oversized = self.client.post(
            "/generate", content=b" " * 8193, headers={"Content-Type": "application/json"}
        )
        self.assertEqual(oversized.status_code, 413)
        invalid = self.client.post("/generate", json={"industry": ""})
        self.assertEqual(invalid.status_code, 422)

    def test_partial_feedback_and_latest_rating(self):
        generated = self.client.post("/generate", json=self.brief()).json()
        generation_id = generated["generation_id"]
        self.assertEqual(len(self.feedback_store.generations[generation_id]["concepts"]), 3)
        first = self.client.post(
            "/feedback",
            json={"generation_id": generation_id, "ratings": [{"concept_index": 1, "rating": "up"}]},
        )
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json()["ratings"], {"1": "up"})
        second = self.client.post(
            "/feedback",
            json={
                "generation_id": generation_id,
                "ratings": [{"concept_index": 1, "rating": "down"}, {"concept_index": 2, "rating": "up"}],
            },
        )
        self.assertEqual(second.json()["ratings"], {"1": "down", "2": "up"})

    def test_feedback_rejects_invalid_and_unknown_generations(self):
        generated = self.client.post("/generate", json=self.brief()).json()
        generation_id = generated["generation_id"]
        invalid = self.client.post(
            "/feedback",
            json={
                "generation_id": generation_id,
                "ratings": [{"concept_index": 0, "rating": "up"}, {"concept_index": 0, "rating": "down"}],
            },
        )
        self.assertEqual(invalid.status_code, 422)
        unknown = self.client.post(
            "/feedback",
            json={
                "generation_id": "00000000-0000-0000-0000-000000000000",
                "ratings": [{"concept_index": 0, "rating": "up"}],
            },
        )
        self.assertEqual(unknown.status_code, 404)

    def test_feedback_storage_failure_is_reported(self):
        class FailingStore(MemoryFeedbackStore):
            def submit(self, generation_id, ratings):
                raise RuntimeError("storage down")

        client = TestClient(create_app(CampaignGenerator(FakeBackend()), FailingStore()))
        try:
            generation_id = client.post("/generate", json=self.brief()).json()["generation_id"]
            response = client.post(
                "/feedback",
                json={"generation_id": generation_id, "ratings": [{"concept_index": 0, "rating": "up"}]},
            )
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["detail"], "feedback_storage_unavailable")
        finally:
            client.close()

    def test_vllm_decoding_requires_schema_and_supplied_channels(self):
        sent = []

        def respond(request):
            sent.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

        class TokenProvider:
            def token(self, audience):
                return "test-token"

        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        backend = VLLMBackend("https://inference.test", "https://inference.test", TokenProvider(), client)
        messages = [
            {"role": "system", "content": "Generate JSON"},
            {"role": "user", "content": json.dumps({"channels": ["LinkedIn", "Email"]})},
        ]
        self.assertEqual(asyncio.run(backend.complete(messages=messages, max_tokens=1536)), "{}")
        schema = sent[0]["guided_json"]
        self.assertIn("channel_plan", schema["required"])
        self.assertIn("campaign_description", schema["required"])
        for item in ("ChannelPlanItem", "AssetPlanItem"):
            self.assertEqual(schema["$defs"][item]["properties"]["channel"]["enum"], ["LinkedIn", "Email"])
        self.assertNotIn("response_format", sent[0])
        asyncio.run(client.aclose())


if __name__ == "__main__":
    unittest.main()
