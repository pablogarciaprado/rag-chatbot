"""
Prompt manager.

Builds the prompt middleware for the RAG system.
"""

from typing import Any, List, NotRequired
from pathlib import Path

from langchain.agents.middleware.types import AgentState

RETRIEVED_DOCS_STATE_KEY = "retrieved_docs"


class RagAgentState(AgentState):
    """
    Agent state extended with chunks retrieved once per query in RagWrapper.
    
    - Agent state: The graph’s working memory during invoke()
    - request.state in middleware: A view of that same state, passed to prompt-building code

    Args:
        retrieved_docs: List[Any] - The retrieved documents.

    Returns:
        RagAgentState - The agent state.

"""

    retrieved_docs: NotRequired[List[Any]]


def build_system_prompt(retrieved_docs: List[Any]) -> str:
    """Build the system prompt, optionally appending retrieved chunk text."""
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    with open(Path(__file__).parent / "system_prompt.txt", "r") as f:
        system_message = f.read()
    system_message += "\n\n"

    if docs_content:
        system_message += (
            " Use the following context in your response:\n\n" + docs_content
        )
    return system_message


def build_prompt_middleware():
    """
    Build the prompt middleware for the RAG system.

    Expects ``retrieved_docs`` to be populated on agent state by ``RagWrapper``
    before ``invoke`` so retrieval runs once per query.

    Returns:
        A prompt middleware function.
    """
    from langchain.agents.middleware import ModelRequest, dynamic_prompt

    @dynamic_prompt
    def prompt_with_context(request: ModelRequest) -> str:
        """
        Callback function that builds the system prompt with the retrieved documents.

        Args:
            request: ModelRequest - The request object.

        Returns:
            str - The system prompt.
        """
        retrieved_docs = request.state.get(RETRIEVED_DOCS_STATE_KEY, [])
        if not retrieved_docs:
            print("No retrieved documents in agent state, continuing without RAG context.")
        return build_system_prompt(retrieved_docs)

    return prompt_with_context
