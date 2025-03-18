import logging
from typing import AsyncGenerator, Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import ChatMessage, TextMessage
from autogen_core import CancellationToken

logger = logging.getLogger(__name__)


class TerminatorAgent(AssistantAgent):
    """
    An agent that always returns 'ΩΩ' without calling the underlying model,
    even though we pass in a legitimate 'model_client'.
    """

    def __init__(self, model_client):
        """
        Pass in a *real* model_client (e.g. OpenAIChatCompletionClient).
        We'll store it in the parent so it looks legit to downstream code,
        but we won't actually invoke it.
        """
        super().__init__(
            name="the_terminator",
            model_client=model_client,
            system_message="Always respond with 'ΩΩ'.",
            reflect_on_tool_use=False,
            tools=[],  # no real tools
            description="Terminates conversation by returning 'ΩΩ' immediately.",
        )

    async def on_messages(
        self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        """
        Non-streaming version. We simply build and return a Response with 'ΩΩ'
        without calling any LLM.
        """
        # Create a single text message
        terminator_msg = TextMessage(content="ΩΩ", source=self.name)
        # Then wrap it in a final Response
        response = Response(
            chat_message=terminator_msg, inner_messages=[terminator_msg]
        )
        return response

    async def on_messages_stream(
        self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[ChatMessage | Response, None]:
        """
        Streaming version. We yield a TextMessage('ΩΩ') and then
        yield a final Response containing the same message.
        """
        # 1) Emit the single text message
        terminator_msg = TextMessage(content="ΩΩ", source=self.name)
        yield terminator_msg

        # 2) Immediately yield a final Response with the same 'ΩΩ'
        yield Response(chat_message=terminator_msg, inner_messages=[terminator_msg])
