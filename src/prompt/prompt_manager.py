"""
Prompt manager.

Builds the prompt middleware for the RAG system.
"""

from typing import Any, List
from pathlib import Path
from langchain_core.vectorstores import InMemoryVectorStore

def _extract_last_query(last_msg: Any) -> str:
    """
    Extract the last query from the messages.
    """
    # Messages returned by LangChain can have either `.content` or `.text`.
    if hasattr(last_msg, "content") and last_msg.content:
        if isinstance(last_msg.content, str):
            return last_msg.content

        # When content is a list of parts, keep only text parts.
        return "".join(
            part.get("text", "")
            for part in last_msg.content
            if isinstance(part, dict)
        )

    if hasattr(last_msg, "text") and last_msg.text:
        return last_msg.text

    return ""

def build_prompt_middleware(vectorstore: InMemoryVectorStore):
    """
    Build the prompt middleware for the RAG system.
    """
    from langchain.agents.middleware import ModelRequest, dynamic_prompt

    @dynamic_prompt
    def prompt_with_context(request: ModelRequest) -> str:
        # Inspect the last user message, do similarity search, and inject retrieved text into the system message.
        messages = request.state["messages"]
        last_msg = messages[-1]
        last_query = _extract_last_query(last_msg)

        retrieved_docs: List[Any] = []
        if last_query.strip():
            try:
                retrieved_docs = vectorstore.similarity_search(last_query)
            except Exception as e:
                # Keep the system message without RAG context.
                print(f"Similarity search failed, continuing without RAG: {e}")

        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # Load system prompt from a file
        with open(Path(__file__).parent / "system_prompt.txt", "r") as f:
            system_message = f.read()
        system_message += "\n\n"

        # Add context if available
        if docs_content:
            system_message += (
                " Use the following context in your response:\n\n" + docs_content
            )
        return system_message

    return prompt_with_context