"""FastAPI public API with a small same-origin browser demo."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import ValidationError

from ..config import (
    API_MAX_BODY_BYTES,
    API_MAX_BRIEF_TEXT_LENGTH,
    API_MAX_CHANNELS,
    API_MAX_CONSTRAINTS,
    API_REJECTED_LOG_LENGTH,
    API_TIMEOUT_SECONDS,
    API_TITLE,
    API_VERSION,
    INFERENCE_AUDIENCE_ENV,
    INFERENCE_URL_ENV,
)
from ..schemas import ApiCampaignBrief
from .backend import GoogleIdentityTokenProvider, UpstreamError, VLLMBackend
from .service import CampaignGenerator

DEMO_HTML = Path(__file__).with_name("demo.html").read_text(encoding="utf-8")
logger = logging.getLogger(__name__)


def create_app(generator: CampaignGenerator | None = None) -> FastAPI:
    """Create app; tests can inject a fake generator without cloud access."""
    backend = None
    injected_generator = generator

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        nonlocal backend, generator
        if generator is None:
            inference_url = os.getenv(INFERENCE_URL_ENV, "").rstrip("/")
            audience = os.getenv(INFERENCE_AUDIENCE_ENV, inference_url).rstrip("/")
            if inference_url and audience:
                backend = VLLMBackend(inference_url, audience, GoogleIdentityTokenProvider())
                generator = CampaignGenerator(backend)
        app.state.generator = generator
        yield
        if backend is not None:
            await backend.close()

    app = FastAPI(title=API_TITLE, version=API_VERSION, lifespan=lifespan)
    app.state.generator = injected_generator

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return DEMO_HTML

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/generate")
    async def generate(request: Request) -> JSONResponse:
        chunks: list[bytes] = []
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > API_MAX_BODY_BYTES:
                raise HTTPException(status_code=413, detail="request_body_too_large")
            chunks.append(chunk)
        try:
            payload: Any = json.loads(b"".join(chunks))
            if not isinstance(payload, dict):
                raise ValueError("request_body_must_be_object")
            brief = ApiCampaignBrief.model_validate(payload)

        except (json.JSONDecodeError, ValueError, ValidationError) as exc:
            raise HTTPException(status_code=422, detail="invalid_campaign_brief") from exc

        if (
            len(payload.get("channels", [])) > API_MAX_CHANNELS
            or len(payload.get("constraints", [])) > API_MAX_CONSTRAINTS
        ):
            raise HTTPException(status_code=422, detail="too_many_channels_or_constraints")

        for key in ("industry", "target_audience", "objective", "brand", "brand_context", "proof_point"):
            value = payload.get(key, "")
            if isinstance(value, str) and len(value) > API_MAX_BRIEF_TEXT_LENGTH:
                raise HTTPException(status_code=422, detail="brief_text_too_long")

        if app.state.generator is None:
            raise HTTPException(status_code=502, detail="inference_not_configured")

        try:
            response = await asyncio.wait_for(
                app.state.generator.generate(brief), timeout=API_TIMEOUT_SECONDS
            )

        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise HTTPException(status_code=504, detail="inference_timeout") from exc

        except (UpstreamError, ValueError) as exc:
            logger.warning("campaign_generation_rejected: %s", str(exc)[:API_REJECTED_LOG_LENGTH])
            raise HTTPException(status_code=502, detail="invalid_model_output") from exc

        except Exception as exc:
            logger.exception("campaign_generation_failed: %s", type(exc).__name__)
            raise HTTPException(status_code=502, detail="inference_failed") from exc

        content = response.model_dump(mode="python")

        for concept in content["concepts"]:
            for allocation in concept["budget_allocation"]:
                amount = allocation["amount"]
                allocation["amount"] = int(amount) if amount == amount.to_integral_value() else float(amount)
        return JSONResponse(content=jsonable_encoder(content))

    return app


app = create_app()
