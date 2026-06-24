from typing import Dict, List, Type

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel

from app.observability.logging import log_llm_parsed, log_llm_request, log_llm_response
from app.services.llm.base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, base_url: str, model: str, temperature: float, max_tokens: int):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.structured_client = instructor.from_openai(self.client)

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        log_llm_request(self.model, messages, temperature=self.temperature, max_tokens=self.max_tokens)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )
        content = response.choices[0].message.content
        log_llm_response(content)
        return content

    async def structured_output(
        self, messages: List[Dict[str, str]], schema: Type[BaseModel], **kwargs
    ) -> BaseModel:
        log_llm_request(self.model, messages, schema=schema.__name__, temperature=self.temperature)
        # instructor handles the raw response + validation internally, so only the parsed
        # result is observable here.
        parsed = await self.structured_client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_model=schema,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )
        log_llm_parsed(schema.__name__, parsed)
        return parsed
