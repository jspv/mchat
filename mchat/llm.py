from typing import List, Callable, Optional, Dict, Set, List, Union, cast, Literal
from dataclasses import dataclass, fields

from pydantic import BaseModel, Field
from pydantic.networks import HttpUrl


# No Dall-e support in langhchain OpenAI, so use the native
from openai import OpenAI

import copy

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import (
    TextMessage,
    AgentMessage,
    ToolCallMessage,
    ToolCallResultMessage,
)
from autogen_agentchat.base import Response, TaskResult
from autogen_core import CancellationToken
from autogen_ext.models import (
    OpenAIChatCompletionClient,
    AzureOpenAIChatCompletionClient,
)
from autogen_core.components.models import (
    FunctionExecutionResultMessage,
    FunctionExecutionResult,
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


@dataclass
class ModelConfig:
    model_id: str
    model: str
    model_type: Literal["chat", "image", "embedding"]
    api_type: Literal["open_ai", "azure"]


@dataclass
class ModelConfigChatOpenAI(ModelConfig):
    api_key: str
    model_type: Literal["chat"]
    api_type: Literal["open_ai"]
    base_url: HttpUrl | None = None
    temperature: float | None = None
    _streaming_support: bool | None = True
    _tool_support: bool | None = True
    _json_support: bool | None = True
    _system_prompt_support: bool | None = True
    _temperature_support: bool | None = None
    _max_tokens: int | None = None
    _max_context: int | None = None
    _cost_input: float | None = None
    _cost_output: float | None = None


@dataclass
class ModelConfigChatAzure(ModelConfig):
    api_key: str
    azure_deployment: str
    api_version: str
    azure_endpoint: HttpUrl
    model_type: Literal["chat"]
    api_type: Literal["azure"]
    temperature: float | None = None
    _streaming_support: bool | None = True
    _tool_support: bool | None = True
    _json_support: bool | None = True
    _system_prompt_support: bool | None = True
    _temperature_support: bool | None = True
    _max_tokens: int | None = None
    _max_context: int | None = None
    _cost_input: float | None = None
    _cost_output: float | None = None


@dataclass
class ModelConfigImageOpenAI(ModelConfig):
    api_key: str
    size: str
    quality: str
    num_images: int
    model_type: Literal["image"]
    api_type: Literal["open_ai"]
    _cost_output: float | None
    _temperature_support: bool | None = None
    _streaming_support: bool | None = False
    _tool_support: bool | None = False
    _system_prompt_support: bool | None = False


@dataclass
class ModelConfigImageAzure(ModelConfig):
    api_key: str
    azure_deployment: str
    api_version: str
    azure_endpoint: HttpUrl
    size: str
    quality: str
    num_images: int
    model_type: Literal["image"]
    api_type: Literal["azure"]
    _cost_ouput: float | None = None
    _temperature_support: bool | None = None
    _streaming_support: bool | None = False
    _tool_support: bool | None = False
    _system_prompt_support: bool | None = False


@dataclass
class ModelConfigEmbeddingOpenAI(ModelConfig):
    model_type: Literal["embedding"]
    api_type: Literal["open_ai"]
    api_key: str
    _cost_input: float | None = None
    _cost_output: float | None = None
    _temperature_support: bool | None = False
    _streaming_support: bool | None = False
    _tool_support: bool | None = False
    _system_prompt_support: bool | None = False


@dataclass
class ModelConfigEmbeddingAzure(ModelConfig):
    model_type: Literal["embedding"]
    api_type: Literal["azure"]
    api_key: str
    azure_deployment: str
    api_version: str
    azure_endpoint: HttpUrl
    _cost_input: float | None = None
    _cost_output: float | None = None
    _temperature_support: bool | None = None
    _streaming_support: bool | None = False
    _tool_support: bool | None = False
    _system_prompt_support: bool | None = False


class ModelManager:
    def __init__(self, logger: Callable | None = None):
        # Load all the models from the settings
        self.model_configs = settings["models"].to_dict()
        self.config: Dict[str, ModelConfig] = {}
        self._logger_callback = logger

        # The constructors for the various model types
        model_types = {
            "chat": {"open_ai": ModelConfigChatOpenAI, "azure": ModelConfigChatAzure},
            "image": {
                "open_ai": ModelConfigImageOpenAI,
                "azure": ModelConfigImageAzure,
            },
            "embedding": {
                "open_ai": ModelConfigEmbeddingOpenAI,
                "azure": ModelConfigEmbeddingAzure,
            },
        }

        # store the model configurations in the appropriate dataclasses
        for model_type in self.model_configs.keys():
            for model_id, model_config in self.model_configs[model_type].items():
                self.config[model_id] = model_types[model_type][
                    model_config["api_type"]
                ](model_id=model_id, model_type=model_type, **model_config)

        # self.chat_config_list = (
        #     [x for x in self.model_configs["chat"].values()]
        #     if self.model_configs.get("chat", None)
        #     else []
        # )
        # self.image_config_list = (
        #     [x for x in self.model_configs["image"].values()]
        #     if self.model_configs.get("image", None)
        #     else []
        # )
        # self.embedding_config_list = (
        #     [x for x in self.model_configs["embedding"].values()]
        #     if self.model_configs.get("embedding", None)
        #     else []
        # )

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
    def logger(self):
        """Returns the current logger"""
        return self._logger_callback

    @logger.setter
    def logger(self, logger: Callable):
        """Set the logger"""
        self._logger_callback = logger

    # if a logger has been provided
    def log(self, message: str):
        """Log a message if a logger has been provided"""
        if self._logger_callback:
            self._logger_callback(message)

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

    # def make_autogen_config_list(self, config_list) -> list:
    #     """Returns a list of available chat models suitable for autogen"""
    #     config_list = copy.deepcopy(config_list)
    #     for config in config_list:
    #         if config.get("azure_endpoint", None):
    #             config["base_url"] = config.pop("azure_endpoint")
    #         if config.get("azure_deployment", None):
    #             config["model"] = config.pop("azure_deployment")
    #         if config.get("api_type", None) and config["api_type"] == "open_ai":
    #             config.pop("api_type")

    #     # remove all keys that start with _, these are internal
    #     config_list = [
    #         {k: v for k, v in x.items() if not k.startswith("_")} for x in config_list
    #     ]
    #     return config_list

    @property
    def autogen_config_list(self) -> list:
        """Returns a list of available chat models suitable for autogen"""
        return self.make_autogen_config_list(self.chat_config_list)

    def filter_config(self, filter_dict):
        """Filter the config list by provider and model.

        Args:
            filter_dict (dict, optional): The filter dict with keys corresponding to a
            field in each config, and values corresponding to lists of acceptable values
            for each key.

        Returns:
            list: The filtered config list.
        """
        return [
            key
            for key, value in self.config.items()
            if all(getattr(value, k) == v for k, v in filter_dict.items())
        ]

    def get_streaming_support(self, model_id: str) -> bool:
        """Returns whether the model supports streaming, assume true"""
        return self.config[model_id]._streaming_support

    def get_tool_support(self, model_id: str) -> bool:
        """Returns whether the model supports tools, assume true"""
        return self.config[model_id]._tool_support

    def get_system_prompt_support(self, model_id: str) -> bool:
        """Returns whether the model supports system prompts, assume true"""
        return self.config[model_id]._system_prompt_support

    def get_temperature_support(self, model_id: str) -> bool:
        """Returns whether the model supports temperature, assume true"""
        return self.config[model_id]._temperature_support

    def open_model(
        self,
        model_id: str,
        **kwargs,
    ) -> object:
        """Opens a model and returns the model object"""

        # Get the model record from the model_configs
        record = copy.deepcopy(self.config[model_id])
        assert record.api_type in ["open_ai", "azure"]

        # strip out the non-model attributes and add the passed in values, whatever
        # remains will be considered kwargs for the model
        model_kwargs = {
            **{
                field.name: getattr(record, field.name)
                for field in fields(record)
                if not field.name.startswith("_")
            },
            **kwargs,
        }
        model_kwargs.pop("api_type")

        if record.model_type == "chat":
            # if temperature is supported and currenlty is None, set to default
            if not getattr(record, "_temperature_support", True):
                model_kwargs.pop("temperature")
            elif getattr(record, "temperature", None) is None:
                model_kwargs.setdefault("temperature", self.default_chat_temperature)

            if record.api_type == "open_ai":
                model = OpenAIChatCompletionClient(**model_kwargs)
            elif record.api_type == "azure":
                model = AzureOpenAIChatCompletionClient(
                    model=model_kwargs["model"],
                    azure_deployment=model_kwargs["azure_deployment"],
                    api_key=model_kwargs["api_key"],
                    api_version=model_kwargs["api_version"],
                    azure_endpoint=model_kwargs["azure_endpoint"],
                )
            return model

        elif record.model_type == "image":
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
    def __init__(
        self,
        message_callback: Callable | None = None,
        personas: dict = {},
        stream_tokens: bool = True,
        logger: Callable | None = None,
    ):
        self.mm = ModelManager()
        self._message_callback = message_callback
        self._personas = personas
        self._stream_tokens = stream_tokens
        self._logger_callback = logger

        # Will be used to cancel ongoing tasks
        self._cancelation_token = None

        # Load LLM inference endpoints from an environment variable or a file
        self.config_list = self.mm.filter_config({"model_type": "chat"})

        if not self.config_list:
            raise ValueError("No chat models found in the configuration")

    @property
    def logger(self):
        """Returns the current logger"""
        return self._logger_callback

    @logger.setter
    def logger(self, logger: Callable):
        """Set the logger"""
        self._logger_callback = logger

    # if a logger has been provided
    def log(self, message: str):
        """Log a message if a logger has been provided"""
        if self._logger_callback:
            self._logger_callback(message)

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
        messages = []
        for m in self.agent._model_context:
            # FunctionExcecutionResult is the content of FunctionExecutionResultMessage
            if isinstance(m, FunctionExecutionResultMessage):
                messages.append(
                    (
                        repr(type(m.content[0])),
                        m.content[0].content,
                        m.content[0].call_id,
                    )
                )
            else:
                messages.append(
                    (
                        repr(type(m)),
                        getattr(m, "content", None),
                        getattr(m, "source", None),
                    )
                )
        return messages

    @property
    def stream_tokens(self) -> bool:
        """Returns the current stream token"""
        return self._stream_tokens

    @stream_tokens.setter
    def stream_tokens(self, value: bool) -> None:
        """Set the stream token"""
        self._stream_tokens = value
        # if there is an agent, set token callback in the agent if the value is True
        if hasattr(self, "agent"):
            if value:
                self.agent._token_callback = self._message_callback
            else:
                self.agent._token_callback = None

    def clear_memory(self) -> None:
        """Clear the memory"""
        # JSP - not sure if going to need this, stubbed out as part of the refactor
        # self.agent.clear_memory()
        if hasattr(self, "agent"):
            self.agent._model_context = []

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
            elif objectname.endswith("FunctionExecutionResult'>"):
                msg_type = FunctionExecutionResultMessage
                msg_args = {
                    "content": [
                        FunctionExecutionResult(call_id=source, content=content)
                    ]
                }
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

        # don't use tools if the model does't support them
        if (
            not self.mm.get_tool_support(model_id)
            or not self.model_client.capabilities["function_calling"]
        ):
            tools = None
        else:
            tools = [google_search_tool]

        # Don't stream if not supported or if disabled by _stream_tokens
        if self.mm.get_streaming_support(model_id) and self._stream_tokens:
            self.stream_tokens = True
            callback = self._message_callback
            self.log(f"token streaming for {model_id} enabled")

        else:
            self.stream_tokens = False
            callback = None
            self.log(f"token streaming for {model_id} disabled or not supported")

        # Don't use system messages if not supported
        if self.mm.get_system_prompt_support(model_id):
            system_message = self.prompt
        else:
            system_message = None

        self.agent = AssistantAgent(
            name=persona,
            model_client=self.model_client,
            tools=tools,
            system_message=system_message,
            token_callback=callback,
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

    async def ask(self, message: str) -> None:
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

        async def team_run_stream(task: str) -> None:
            # Generate a cancelation token
            self._cancelation_token = CancellationToken()

            # Clear the termination condition
            # self.agent_team._termination_condition.reset()

            async for response in self.agent_team.run_stream(
                task=message, cancellation_token=self._cancelation_token
            ):
                if isinstance(response, TextMessage):
                    if response.source == "user":
                        # ignore these, it is a repeat of the user message
                        continue
                    elif self.stream_tokens:
                        # ingore these, they are tokens which are being streamed
                        continue
                    else:
                        # not streaming, so send the response
                        await self._message_callback(response.content)
                        continue
                if isinstance(response, ToolCallMessage):
                    await self._message_callback(
                        f"calling {response.content[0].name}..."
                    )
                    continue
                if isinstance(response, ToolCallResultMessage):
                    await self._message_callback("done\n\n")
                    continue
                if isinstance(response, TaskResult):
                    return TaskResult
                if response is None:
                    continue
                await self._message_callback("\n\n<unknown>\n\n", flush=True)
                await self._message_callback(repr(response), flush=True)

        # tokens are returned via the callback
        result = await team_run_stream(task=message)
        await self._message_callback(f"\n\nresult={result.stop_reason}\n✅", flush=True)


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
