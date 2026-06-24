from abc import ABC, abstractmethod
from typing import Dict, List, Type

from pydantic import BaseModel


class BaseLLMProvider(ABC):
    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str: ...

    @abstractmethod
    async def structured_output(
        self, messages: List[Dict[str, str]], schema: Type[BaseModel], **kwargs
    ) -> BaseModel: ...
