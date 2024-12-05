from typing import List, Callable, Optional, Dict, Set, List, Union, cast

# No Dall-e support in langhchain OpenAI, so use the native
from openai import OpenAI

import copy

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage, ToolCallMessage
from autogen_agentchat.base import Response
from autogen_core import CancellationToken
from autogen_ext.models import (
    OpenAIChatCompletionClient,
    AzureOpenAIChatCompletionClient,
)
from autogen_core.components.models import (
    FunctionExecutionResultMessage,
    LLMMessage,
    SystemMessage,
    UserMessage,
    AssistantMessage,
)
from autogen_core.components.tools import FunctionTool
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat

from config import settings
from functools import partial
import asyncio

import logging
import http.client

from .llmtool_web_search import google_search

logging.basicConfig(filename="debug.log", filemode="w", level=logging.DEBUG)
requests_log = logging.getLogger("requests.packages.urllib3")
requests_log.setLevel(logging.DEBUG)
requests_log.propagate = True
http.client.HTTPConnection.debuglevel = 1

intent_prompt = (
    "A human is having a conversation with an AI, you are watching that conversation"
    " and your job is to determine what the human is intending to do in their last"
    " message to the AI.  Reply with one of the following options:  1) if the human is"
    " asking to change their persona, reply with the word 'persona', 2) if the human is"
    " asking a new question unrelated to the current conversation, reply with the word"
    " 'ask', 3) if the human is continuing an existing conversation, reply with the"
    " word 'continue' 4) if the human is responding to a question that the AI asked,"
    " reply with the word 'reply', 5) if the human is trying to open or loading a file,"
    " reply with the word 'open', 6) if the human is trying to start a new session,"
    " reply with the word 'new', 7) if the human is trying to end the conversation,"
    " reply with the word 'end'.  The previous conversation follows: {conversation}\n"
    " The new message to the AI is: {input}"
)

label_prompt = (
    "Here is a conversation between a Human and AI. Give me no more than 10 words which"
    " will be used to remind the user of the conversation.  Your reply should be no"
    " more than 10 words and at most 66 total characters. Here is the conversation"
    " {conversation}"
)

summary_prompt = (
    "Here is a conversation between a Human and AI. Provide a detailed summary of the "
    "Conversation: "
    "{conversation}"
)


class ModelManager:
    def __init__(self):
        # Load all the models from the settings
        self.model_configs = settings["models"].to_dict()
        self.chat_config_list = (
            [x for x in self.model_configs["chat"].values()]
            if self.model_configs.get("chat", None)
            else []
        )
        self.image_config_list = (
            [x for x in self.model_configs["image"].values()]
            if self.model_configs.get("image", None)
            else []
        )
        self.embedding_config_list = (
            [x for x in self.model_configs["embedding"].values()]
            if self.model_configs.get("embedding", None)
            else []
        )

        # Set the default models
        self.default_chat_model = settings.defaults.chat_model
        self.default_image_model = settings.defaults.image_model
        self.default_embedding_model = settings.defaults.embedding_model
        self.default_chat_temperature = settings.defaults.chat_temperature
        self.default_memory_model = settings.defaults.memory_model
        self.default_memory_model_temperature = (
            settings.defaults.memory_model_temperature
        )
        self.default_memory_model_max_tokens = settings.defaults.memory_model_max_tokens

        # os.environ["OAI_CONFIG_LIST"] = json.dumps(self.chat_config_list)

    @property
    def available_image_models(self) -> list:
        """Returns a list of available image models"""
        return [x for x in self.model_configs["image"].keys()]

    @property
    def available_chat_models(self) -> list:
        """Returns a list of available chat models"""
        return [x for x in self.model_configs["chat"].keys()]

    @property
    def available_embedding_models(self) -> list:
        """Returns a list of available embedding models"""
        return [x for x in self.model_configs["embedding"].keys()]

    def make_autogen_config_list(self, config_list) -> list:
        """Returns a list of available chat models suitable for autogen"""
        config_list = copy.deepcopy(config_list)
        for config in config_list:
            if config.get("azure_endpoint", None):
                config["base_url"] = config.pop("azure_endpoint")
            if config.get("azure_deployment", None):
                config["model"] = config.pop("azure_deployment")
            if config.get("api_type", None) and config["api_type"] == "open_ai":
                config.pop("api_type")

        # remove all keys that start with _, these are internal
        config_list = [
            {k: v for k, v in x.items() if not k.startswith("_")} for x in config_list
        ]
        return config_list

    @property
    def autogen_config_list(self) -> list:
        """Returns a list of available chat models suitable for autogen"""
        return self.make_autogen_config_list(self.chat_config_list)

    @staticmethod
    def filter_config(config_list, filter_dict):
        """Filter the config list by provider and model.

        Args:
            config_list (list): The config list.
            filter_dict (dict, optional): The filter dict with keys corresponding to a
            field in each config, and values corresponding to lists of acceptable values
            for each key.

        Returns:
            list: The filtered config list.
        """
        if filter_dict:
            config_list = [
                config
                for config in config_list
                if all(config.get(key) in value for key, value in filter_dict.items())
            ]
        return config_list

    def open_model(
        self,
        model_id: str,
        model_type="chat",
        **kwargs,
    ) -> object:
        """Opens a model and returns the model object"""

        # Get the model record from the model_configs
        record = copy.deepcopy(self.model_configs[model_type][model_id])
        assert record["api_type"] in ["open_ai", "azure"]

        # strip out the non-model attributes and add the passed in values
        model_kwargs = {
            **{k: v for k, v in record.items() if not k.startswith("_")},
            **kwargs,
        }

        if model_type == "chat":
            model_kwargs.setdefault("temperature", self.default_chat_temperature)
            model_kwargs.pop("api_type")

            if record["api_type"] == "open_ai":
                model = OpenAIChatCompletionClient(
                    model=model_id, api_key=model_kwargs["api_key"]
                )
            elif record["api_type"] == "azure":
                model = AzureOpenAIChatCompletionClient(
                    model=model_kwargs["model"],
                    azure_deployment=model_kwargs["azure_deployment"],
                    api_key=model_kwargs["api_key"],
                    api_version=model_kwargs["api_version"],
                    azure_endpoint=model_kwargs["azure_endpoint"],
                )
            return model

        elif model_type == "image":
            model = DallEAPIWrapper(**model_kwargs)
            return model
        else:
            raise ValueError("Invalid model_type")


class DallEAPIWrapper(object):
    """Wrapper for OpenAI's DALL-E Image Generator.

    https://platform.openai.com/docs/guides/images/generations?context=node

    """

    def __init__(
        self,
        api_key: str | None = None,
        num_images: int = 1,
        size: str = "1024x1024",
        separator: str = "\n",
        model: str | None = "dall-e-3",
        quality: str | None = "standard",
        **kwargs,
    ):
        self.api_key = api_key
        self.num_images = num_images
        self.size = size
        self.separator = separator
        self.model = model
        self.quality = quality

        self.client = OpenAI(api_key=self.api_key)

    def run(self, query: str) -> str:
        """Run query through OpenAI and parse result."""
        response = self.client.images.generate(
            prompt=query,
            n=self.num_images,
            size=self.size,
            model=self.model,
            quality=self.quality,
        )
        image_urls = self.separator.join([item.url for item in response.data])
        return image_urls if image_urls else "No image was generated"

    async def arun(self, query: str) -> str:
        """Run query through OpenAI and parse result."""
        loop = asyncio.get_running_loop()
        # prepare keyword arguments for the run_in_executor call by partial
        sync_func = partial(
            self.client.images.generate,
            prompt=query,
            n=self.num_images,
            size=self.size,
            model=self.model,
            quality=self.quality,
        )
        response = await loop.run_in_executor(None, sync_func)
        image_urls = self.separator.join([item.url for item in response.data])
        return image_urls if image_urls else "No image was generated"


class AutogenManager(object):
    def __init__(self, message_callback: Callable | None = None, personas: dict = {}):
        self.mm = ModelManager()
        self._message_callback = message_callback
        self._personas = personas

        # Load LLM inference endpoints from an environment variable or a file
        self.config_list = self.mm.make_autogen_config_list(
            self.mm.filter_config(
                self.mm.chat_config_list, {"api_type": "open_ai", "model": "gpt-4o"}
            )
            # self.mm.filter_config(self.mm.chat_config_list, {})
        )

        if not self.config_list:
            raise ValueError("No chat models found in the configuration")

    def new_agent(self, agent_name, model_name, prompt, tools=[]) -> None:
        """Create a new agent with the given name, model, tools, and prompt"""
        pass

    @property
    def prompt(self) -> str:
        """Returns the current prompt"""
        # return self.agent.prompt
        return self._prompt

    @property
    def memory(self) -> str:
        """Returns the current memory in a recoverable text format"""
        # (class, content, source)
        return [
            (
                repr(type(m)),
                getattr(m, "content", None),
                getattr(m, "source", None),
            )
            for m in self.agent._model_context
        ]

    def clear_memory(self) -> None:
        """Clear the memory"""
        # JSP - not sure if going to need this, stubbed out as part of the refactor
        # self.agent.clear_memory()
        self.agent._model_context: List[LLMMessage] = []

    def update_memory(self, messages: List[str]) -> None:
        """Update the memory witt the contents of the messages"""

        self.clear_memory()

        # Parse the messages and add them to the memory, we're expecting them
        # to be in the format of a list of strings, where each string is the
        # tuple representation of a message object, content, and source

        for objectname, content, source in messages:
            if objectname.endswith("UserMessage'>"):
                msg_type = UserMessage
                msg_args = {"content": content, "source": source}
            elif objectname.endswith("AssistantMessage'>"):
                msg_type = AssistantMessage
                msg_args = {"content": content, "source": source}
            elif objectname.endswith("SystemMessage.>"):
                msg_type = SystemMessage
                msg_args = {"content": content, "source": source}
            elif objectname.endswith("ToolCallMessage'>"):
                msg_type = ToolCallMessage
                msg_args = {"content": content, "source": source}
            elif objectname.endswith("FunctionExecutionResultMessage'>"):
                msg_type = FunctionExecutionResultMessage
                msg_args = {"content": content}

            else:
                raise ValueError(f"Unexpected message type: {objectname}")

            # Add the appropriate object to the memory
            self.agent._model_context.append(msg_type(**msg_args))

    def new_conversation(
        self, persona: dict, model_id, temperature: float = 0.0
    ) -> None:
        """Intialize a new conversation with the given persona and model

        Parameters
        ----------
        persona : dict
            persona object
        model_id: str
            model  to use""
        temperature : float, optional
            model temperature setting, by default 0.0
        """

        self.model_client = self.mm.open_model(model_id)

        self._prompt = self._personas[persona]["description"]

        google_search_tool = FunctionTool(
            google_search,
            description="Search Google for information, returns results with a snippet and body content",
        )

        self.agent = AssistantAgent(
            name=persona,
            model_client=self.model_client,
            tools=[google_search_tool],
            system_message=self.prompt,
            token_callback=self._message_callback,
        )

        # Load Extra system messages
        for extra in self._personas[persona]["extra_context"]:
            if extra[0] == "ai":
                self.agent._model_context.append(
                    AssistantMessage(content=extra[1], source=persona)
                )
            elif extra[0] == "human":
                self.agent._model_context.append(
                    UserMessage(content=extra[1], source="user")
                )
            elif extra[0] == "system":
                raise ValueError(f"system message not implemented: {extra[0]}")
            else:
                raise ValueError(f"Unknown extra context type {extra[0]}")

        termination = MaxMessageTermination(max_messages=2)
        self.agent_team = RoundRobinGroupChat(
            [self.agent], termination_condition=termination
        )

    async def ask(self, message: str):
        async def assistant_run_stream() -> None:
            async for response in self.agent.on_messages_stream(
                [TextMessage(content=message, source="user")],
                cancellation_token=CancellationToken(),
            ):
                if isinstance(response, Response) and isinstance(
                    response.chat_message, TextMessage
                ):
                    await self._message_callback(
                        response.chat_message.content, flush=True
                    )

        async def assistant_run() -> None:
            response = await self.agent.on_messages(
                [TextMessage(content=message, source="user")],
                cancellation_token=CancellationToken(),
            )

        # tokens are returned via the callback
        result = await self.agent_team.run(task=message)
        # await callback("done with run", flush=True)
        # await callback(repr(result), flush=True)


class LLMTools:
    def __init__(self):
        """Builds the intent prompt template"""

        self.mm = ModelManager()

        # # build the intent chain
        # self.intent_chain = (
        #     ChatPromptTemplate.from_template(intent_prompt)
        #     | self.mm.open_model(self.mm.default_memory_model)
        #     | StrOutputParser()
        # )

        # # build the summary chain
        # self.summary_label_chain = (
        #     ChatPromptTemplate.from_template(label_prompt)
        #     | self.mm.open_model(self.mm.default_memory_model)
        #     | StrOutputParser()
        # )

    async def aget_intent(self, question: str, memory) -> str:
        """Returns the intent of the input"""

        # get up to the last 3 turns of the conversation from memory
        history = memory.load_memory_variables({})["history"]
        conversation = self._parse_chat_history(history, max_turns=3)

        out = await self.intent_chain.ainvoke(
            {"conversation": conversation, "input": question}
        )
        return out

    def get_intent(self, question: str, memory) -> str:
        """Returns the intent of the input"""

        # get up to the last 3 turns of the conversation from memory
        history = memory.load_memory_variables({})["history"]
        conversation = self._parse_chat_history(history, max_turns=3)

        out = self.intent_chain.invoke(
            {"conversation": conversation, "input": question}
        )
        return out

    @staticmethod
    async def aget_summary_label(conversation: str) -> str:
        """Returns the a very short summary of a conversation suitable for a label"""
        mm = ModelManager()
        model_client = mm.open_model(mm.default_memory_model)
        system_message = label_prompt.format(conversation=conversation)
        out = await model_client.create([SystemMessage(content=system_message)])
        return out.content

    @staticmethod
    async def get_conversation_summary(conversation: str) -> str:
        """Returns the summary of a conversation"""
        mm = ModelManager()
        model_client = mm.open_model(mm.default_memory_model)
        system_message = summary_prompt.format(conversation=conversation)
        out = await model_client.create([SystemMessage(content=system_message)])
        return out.content

    def _parse_chat_history(self, history: list, max_turns: int | None = None) -> list:
        """Parses the chat history into a list of conversation turns"""
        if len(history) == 0:
            return "There is no conversation history, this is the first message"

        if max_turns:
            history = history[-max_turns:]

        chat_history = []

        for message in history:
            content = cast(str, message.content)
            if isinstance(message, HumanMessage):
                chat_history.append({f'"Human": {content}'})
            if isinstance(message, AIMessage):
                chat_history.append({f'"AI: {content}'})
        return chat_history
