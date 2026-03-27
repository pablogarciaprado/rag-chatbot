from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseLLMProvider(ABC):
    """
    Base class for configuring (not custom-competing) LLM providers.

    Subclasses must define a non-empty `model` string at the class level:
      class MyProvider(BaseLLMProvider):
          model = "some-model"

    `rag.py` (via this class) uses `self.model` when instantiating the actual
    LangChain LLM passed to `create_agent(...)`.
    """

    model: str
    temperature: float = 0.0

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not isinstance(getattr(cls, "model", None), str) or not cls.model.strip():
            raise TypeError(
                f"{cls.__name__}.model must be a non-empty string (got {getattr(cls, 'model', None)!r})."
            )
    
    @abstractmethod
    def build_llm(self) -> Any:
        """
        Instantiate the concrete LangChain LLM using the provider config.
        """
        pass


