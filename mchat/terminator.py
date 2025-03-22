import logging
import traceback
from collections.abc import Sequence

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import (
    ChatMessage,
    StopMessage,
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallSummaryMessage,
)
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient, SystemMessage

logger = logging.getLogger(__name__)


class SmartReflectorAgent(BaseChatAgent):
    """
    An agent that always returns StopMessage without calling the underlying model"""

    terminator_prompt = (
        "Below is a conversation between 'user' and '{agent}'. "
        "Look at the user's statement and the last message from {agent} and determine "
        "if {agent} is still working on responding to the user.  Follow these instructions:  "
        "1) If {agent} is asking a question to user or is asking for input, respond with 'END'"
        "2) If {agent} is still working on a response and has another step to take, reply with '{agent}"
        "3) Otherise, reply with 'END'. "
        "Here is the conversation: {history}"
    )

    def __init__(
        self,
        model_client: ChatCompletionClient,
        name: str = "the_terminator",
        agent_name: str = "ai",
        oneshot: bool = True,
    ):
        """
        Pass in a *real* model_client (e.g. OpenAIChatCompletionClient).
        We'll store it in the parent so it looks legit to downstream code,
        but we won't actually invoke it.
        """
        super().__init__(
            name=name,
            description="Terminates conversation",
        )
        self.agent_name = agent_name
        self.oneshot = oneshot
        self.model_client = model_client
        self._message_history: list[ChatMessage] = []

    async def on_messages(
        self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        """
        Return a StopMessage Response with without calling any LLM.
        """
        self._message_history.extend(messages)
        if self.oneshot:
            return Response(chat_message=StopMessage(content="done", source=self.name))

        last_message = self._message_history[-1]
        # last message was from the user, likely should not occur
        if isinstance(last_message, TextMessage) and last_message.source == "user":
            return Response(chat_message=TextMessage(content="", source=self.name))

        # this will force a reflection on the tool call
        if isinstance(last_message, ToolCallSummaryMessage | ToolCallExecutionEvent):
            return Response(chat_message=TextMessage(content="", source=self.name))

        # get up to the last 6 messages
        history = ""
        for message in self._message_history[-6:]:
            history += f"{message.source}: {message.content}\n"
        context = [
            SystemMessage(
                content=self.terminator_prompt.format(
                    agent=self.agent_name, history=history
                )
            )
        ]

        try:
            result = await self.model_client.create(
                messages=context,
                cancellation_token=cancellation_token,
            )
            if result.content == "END":
                return Response(
                    chat_message=StopMessage(content="done", source=self.name)
                )
            else:
                logger.debug(f"reflecting back: {result.content}")
                return Response(chat_message=TextMessage(content="", source=self.name))

        except Exception as e:
            logger.error(f"Error from model client: {e}")
            traceback.print_exc()
            return Response(chat_message=StopMessage(content="error", source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

    @property
    def produced_message_types(self) -> Sequence[type[ChatMessage]]:
        return (StopMessage, TextMessage)


class StopTerminatorAgent(BaseChatAgent):
    """
    An agent that always returns StopMessage without calling the underlying model"""

    def __init__(
        self,
        name: str = "the_terminator",
    ):
        super().__init__(
            name=name,
            description="Terminates conversation",
        )

    async def on_messages(
        self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        """
        Return a StopMessage Response without calling any LLM.
        """
        return Response(chat_message=StopMessage(content="done", source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

    @property
    def produced_message_types(self) -> Sequence[type[ChatMessage]]:
        return (StopMessage,)
