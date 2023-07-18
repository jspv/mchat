import asyncio
from typing import Any, Dict, List

from langchain.chat_models import ChatOpenAI
from langchain.schema import LLMResult, HumanMessage
from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
from langchain.chains import ConversationChain
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chains.conversation.memory import ConversationSummaryBufferMemory


class MyCustomSyncHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"Sync handler being called in a `thread_pool_executor`: token: {token}")


class MyCustomAsyncHandler(AsyncCallbackHandler):
    """Async callback handler that can be used to handle callbacks from langchain."""

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""
        print("zzzz....")
        await asyncio.sleep(0.3)
        #        class_name = serialized["name"]
        print("Hi! I just woke up. Your llm is starting")

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"async: {token}")

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when chain ends running."""
        print("zzzz....")
        await asyncio.sleep(0.3)
        print("Hi! I just woke up. Your llm is ending")


memory = ConversationSummaryBufferMemory(
    llm=ChatOpenAI(), max_token_limit=1000, return_messages=True
)

prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)

# To enable streaming, we pass in `streaming=True` to the ChatModel constructor
# Additionally, we pass in a list with our custom handler
chat = ChatOpenAI(
    max_tokens=25,
    streaming=True,
    #    callbacks=[MyCustomSyncHandler(), MyCustomAsyncHandler()],
    callbacks=[MyCustomAsyncHandler()],
)

conversation = ConversationChain(
    llm=chat,
    memory=memory,
    prompt=prompt,
    verbose=True,
)


async def main():
    # await chat.agenerate([[HumanMessage(content="Tell me dog a joke")]])
    await conversation.arun("Tell me dog a joke")


asyncio.run(main())
