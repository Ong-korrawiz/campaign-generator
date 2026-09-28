"""Authenticated async HTTP client for the private Cloud Run vLLM service."""

from __future__ import annotations

import copy
import json
from typing import Protocol

import httpx
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import id_token

from ..config import INFERENCE_MODEL_NAME, INFERENCE_TEMPERATURE, INFERENCE_TIMEOUT_SECONDS
from ..schemas import CampaignDirectionV2


class TokenProvider(Protocol):
    def token(self, audience: str) -> str: ...


class GoogleIdentityTokenProvider:
    """Mint an identity token with the exact Cloud Run service URL as audience."""

    def __init__(self, fetcher=None) -> None:
        self._fetcher = fetcher or id_token.fetch_id_token

    def token(self, audience: str) -> str:
        return self._fetcher(GoogleAuthRequest(), audience)


class UpstreamError(RuntimeError):
    """Private inference returned a non-success response or invalid payload."""


class VLLMBackend:
    """Send OpenAI-compatible chat completions to private vLLM."""

    def __init__(
        self,
        inference_url: str,
        audience: str,
        token_provider: TokenProvider,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = inference_url.rstrip("/") + "/v1/chat/completions"
        self._audience = audience
        self._tokens = token_provider
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(INFERENCE_TIMEOUT_SECONDS))
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def complete(self, *, messages: list[dict[str, str]], max_tokens: int) -> str:
        token = self._tokens.token(self._audience)
        # vLLM 0.8.5 supports guided_json. Constrain required fields and channel
        # values during decoding, then keep Pydantic and cross-field checks in
        # CampaignGenerator as the final validation boundary.
        schema = copy.deepcopy(CampaignDirectionV2.model_json_schema())
        brief = json.loads(messages[1]["content"])
        channels = brief["channels"]
        for item in ("ChannelPlanItem", "AssetPlanItem"):
            schema["$defs"][item]["properties"]["channel"]["enum"] = channels
        try:
            response = await self._client.post(
                self._url,
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "model": INFERENCE_MODEL_NAME,
                    "messages": messages,
                    "temperature": INFERENCE_TEMPERATURE,
                    "max_tokens": max_tokens,
                    "guided_json": schema,
                },
            )
        except httpx.TimeoutException as exc:
            raise TimeoutError("inference_timeout") from exc
        except httpx.HTTPError as exc:
            raise UpstreamError("inference_unavailable") from exc
        if response.status_code >= 500:
            raise UpstreamError("inference_server_error")
        if response.status_code >= 400:
            raise UpstreamError("inference_request_rejected")
        try:
            payload = response.json()
            return payload["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise UpstreamError("invalid_inference_response") from exc
