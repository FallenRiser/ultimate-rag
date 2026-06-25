from typing import Dict, List, Type

from langchain_core.messages import convert_to_messages
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.observability.logging import log_llm_parsed, log_llm_request, log_llm_response
from app.services.llm.base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """OpenAI / vLLM (OpenAI-compatible) via langchain-openai."""

    def __init__(self, api_key: str, base_url: str, model: str, temperature: float, max_tokens: int):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = ChatOpenAI(
            model=model,
            api_key=api_key or None,   # None → reads OPENAI_API_KEY env
            base_url=base_url or None,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        log_llm_request(self.model, messages, temperature=self.temperature, max_tokens=self.max_tokens)
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
