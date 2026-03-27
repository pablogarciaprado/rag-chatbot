from .base import BaseLLMProvider
from langchain_google_genai import ChatGoogleGenerativeAI

class GeminiFlashLiteProvider(BaseLLMProvider):
    model = "gemini-2.5-flash-lite"
    temperature = 0.0 # deterministic output

    def build_llm(self) -> ChatGoogleGenerativeAI:
        """
        Instantiate the concrete LangChain LLM using the provider config.
        """
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=self.model,
            temperature=self.temperature,
        )