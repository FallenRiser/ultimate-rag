from typing import Dict, List, Type

import httpx
from pydantic import BaseModel

from app.observability.logging import log_llm_parsed, log_llm_request, log_llm_response
from app.services.llm.base import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    def __init__(self, base_url: str, model: str, temperature: float, max_tokens: int):
        # Native Ollama API lives at the root (/api/chat), not under /v1. Tolerate a
        # /v1 suffix so the same base_url works for both OpenAI-compat and native.
        base = base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")].rstrip("/")
        self.base_url = base
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        log_llm_request(self.model, messages, temperature=self.temperature)
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": messages, "stream": False},
            )
            response.raise_for_status()
            content = response.json()["message"]["content"]
        log_llm_response(content)
        return content

    async def structured_output(
        self, messages: List[Dict[str, str]], schema: Type[BaseModel], **kwargs
    ) -> BaseModel:
        # Ollama constrains generation to a JSON schema via the `format` parameter.
        log_llm_request(self.model, messages, schema=schema.__name__, temperature=self.temperature)
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "format": schema.model_json_schema(),
                    "options": {"temperature": self.temperature},
                },
            )
            response.raise_for_status()
            content = response.json()["message"]["content"]
        log_llm_response(content)
        parsed = schema.model_validate_json(content)
        log_llm_parsed(schema.__name__, parsed)
        return parsed
