from typing import Dict, List, Type

from langchain_core.messages import convert_to_messages
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from app.observability.logging import log_llm_parsed, log_llm_request, log_llm_response
from app.services.llm.base import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    """Local Ollama models via langchain-ollama (native API)."""

    def __init__(self, base_url: str, model: str, temperature: float, max_tokens: int):
        # langchain-ollama talks to the native API root; tolerate a /v1 suffix in config.
        base = base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")].rstrip("/")
        self.model = model
        self.temperature = temperature
        self.client = ChatOllama(
            model=model, base_url=base, temperature=temperature, num_predict=max_tokens
        )

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        log_llm_request(self.model, messages, temperature=self.temperature)
        response = await self.client.ainvoke(convert_to_messages(messages))
        log_llm_response(response.content)
        return response.content

    async def structured_output(
        self, messages: List[Dict[str, str]], schema: Type[BaseModel], **kwargs
    ) -> BaseModel:
        log_llm_request(self.model, messages, schema=schema.__name__, temperature=self.temperature)
        structured_client = self.client.with_structured_output(schema)
        parsed = await structured_client.ainvoke(convert_to_messages(messages))
        log_llm_parsed(schema.__name__, parsed)
        return parsed
