import asyncio
import logging
from typing import AsyncGenerator, Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import ChatMessage, StopMessage
from autogen_core import CancellationToken

logger = logging.getLogger(__name__)


class TerminatorAgent(AssistantAgent):
    """
    An agent that always returns 'ΩΩ' without calling the underlying model,
    even though we pass in a legitimate 'model_client'.
    """

    def __init__(self, model_client, name="the_terminator", agent_name="ai"):
        """
        Pass in a *real* model_client (e.g. OpenAIChatCompletionClient).
        We'll store it in the parent so it looks legit to downstream code,
        but we won't actually invoke it.
        """
        super().__init__(
            name=name,
            model_client=model_client,
            system_message="not used",
            reflect_on_tool_use=False,
            tools=[],  # no real tools
            description="Terminates conversation",
        )
        self.agent_name = agent_name

    async def on_messages(
        self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        """
        Non-streaming version. We simply build and return a Response with 'ΩΩ'
        without calling any LLM.
        """
        asyncio.sleep(0.2)
        return Response(
            chat_message=StopMessage(content="done", source=self.agent_name)
        )

    async def on_messages_stream(
        self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[ChatMessage | Response, None]:
        """
        Streaming version. We yield a TextMessage('ΩΩ') and then
        yield a final Response containing the same message.
        """
        yield Response(chat_message=StopMessage(content="done", source=self.agent_name))
