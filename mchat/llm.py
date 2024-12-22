import asyncio
import copy
import http.client
import logging
from dataclasses import dataclass, fields
from functools import partial
from typing import (
    AsyncIterable,
    Callable,
    Dict,
    Literal,
)

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import (
    ExternalTermination,
    MaxMessageTermination,
    TextMentionTermination,
)
from autogen_agentchat.messages import (
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
)
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_core.models import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import (
    AzureOpenAIChatCompletionClient,
    OpenAIChatCompletionClient,
)

# No Dall-e support in langhchain OpenAI, so use the native
from openai import OpenAI
from pydantic.networks import HttpUrl

from config import settings

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
    " asking to change their agent, reply with the word 'agent', 2) if the human is"
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
        agents: dict | None = None,
        stream_tokens: bool = True,
        logger: Callable | None = None,
    ):
        self.mm = ModelManager()
        self._message_callback = message_callback
        self._agents = agents if agents is not None else {}
        self._stream_tokens = stream_tokens
        self._logger_callback = logger

        # model being used by the agent, will be set when a new conversation is started
        self._model_id = None

        # Will be used to cancel ongoing tasks
        self._cancelation_token = None

        # Load LLM inference endpoints from an environment variable or a file
        self.config_list = self.mm.filter_config({"model_type": "chat"})

        if not self.config_list:
            raise ValueError("No chat models found in the configuration")

        # Inialize available tools
        self.tools = {}
        self.tools["google_search"] = FunctionTool(
            google_search,
            description=(
                "Search Google for information, returns results with a snippet and "
                "body content"
            ),
        )

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

    def new_agent(
        self, agent_name, model_name, prompt, tools: list | None = None
    ) -> None:
        """Create a new agent with the given name, model, tools, and prompt"""
        pass

    @property
    def prompt(self) -> str:
        """Returns the current prompt"""
        # return self.agent.prompt
        return self._prompt

    async def get_memory(self) -> str:
        """Returns the current memory in a recoverable text format"""
        # save_state is async, so we need to await it
        return await self.agent.save_state()

    @property
    def stream_tokens(self) -> bool:
        """Are we currently streaming tokens if they are supported"""
        return self._stream_tokens

    @stream_tokens.setter
    def stream_tokens(self, value: bool) -> None:
        """enable or disable token streaming"""
        self._stream_tokens = value
        # if there is an agent, set token callback in the agent if the value is True
        if hasattr(self, "agent") and self.mm.get_streaming_support(self._model_id):
            if value:
                self.agent._token_callback = self._message_callback
                self._currently_streaming = True
            else:
                self.agent._token_callback = None
                self._currently_streaming = False

    def clear_memory(self) -> None:
        """Clear the memory"""
        # JSP - not sure if going to need this, stubbed out as part of the refactor
        # self.agent.clear_memory()
        if hasattr(self, "agent"):
            self.agent._model_context.clear()

    async def update_memory(self, state: dict) -> None:
        await self.agent.load_state(state)

    # def old_update_memory(self, state: dict) -> None:
    #     """Update the memory witt the contents of the messages"""

    #     self.clear_memory()

    #     # Parse the messages and add them to the memory, we're expecting them
    #     # to be in the format of a list of strings, where each string is the
    #     # tuple representation of a message object, content, and source

    #     for objectname, content, source in messages:
    #         if objectname.endswith("UserMessage'>"):
    #             msg_type = UserMessage
    #             msg_args = {"content": content, "source": source}
    #         elif objectname.endswith("AssistantMessage'>"):
    #             msg_type = AssistantMessage
    #             msg_args = {"content": content, "source": source}
    #         elif objectname.endswith("SystemMessage.>"):
    #             msg_type = SystemMessage
    #             msg_args = {"content": content, "source": source}
    #         elif objectname.endswith("ToolCallMessage'>"):
    #             msg_type = ToolCallMessage
    #             msg_args = {"content": content, "source": source}
    #         elif objectname.endswith("FunctionExecutionResult'>"):
    #             msg_type = FunctionExecutionResultMessage
    #             msg_args = {
    #                 "content": [
    #                     FunctionExecutionResult(call_id=source, content=content)
    #                 ]
    #             }
    #         else:
    #             raise ValueError(f"Unexpected message type: {objectname}")

    #         # Add the appropriate object to the memory
    #         self.agent._model_context.append(msg_type(**msg_args))

    def new_conversation(self, agent: dict, model_id, temperature: float = 0.0) -> None:
        """Intialize a new conversation with the given agent and model

        Parameters
        ----------
        agent : dict
            agent object
        model_id: str
            model  to use""
        temperature : float, optional
            model temperature setting, by default 0.0
        """

        """
        Thoughts:  model_id and temperature will be for the agent that is interacting
        with the user.  For single-model agents, this will be the same as going straight
        to the model.

        For multi-agent teams, it will only affect the agent that is currently
        interacting with the user.  The other agents in the team will have their own
        model_id and temperature settings from the agent dictionary, or will use the
        default.
        """

        agent_data = self._agents[agent]

        if "type" in agent_data and agent_data["type"] == "team":
            if agent_data["team_type"] == "round_robin":
                self.agent_team = self._create_team_round_robin(agent_data)
                self._prompt = agent_data["description"]
                self.agent = self.agent_team._participants[0]
                self._currently_streaming = False
        else:
            self.model_client = self.mm.open_model(model_id)
            self._model_id = model_id

            self._prompt = self._agents[agent]["description"]

            # don't use tools if the model does't support them
            if (
                not self.mm.get_tool_support(model_id)
                or not self.model_client.capabilities["function_calling"]
            ):
                tools = None
            else:
                tools = [self.tools["google_search"]]

            # wrap the callback in a partial to pass the agent name
            callback = partial(self._message_callback, agent=agent)

            # Stream if supported and enabled by  _stream_tokens
            if self.mm.get_streaming_support(model_id) and self._stream_tokens:
                callback = callback
                self.log(f"token streaming for {model_id} enabled")
                self._currently_streaming = True
            else:
                callback = None
                self.log(f"token streaming for {model_id} disabled or not supported")
                self._currently_streaming = False

            # Don't use system messages if not supported
            if self.mm.get_system_prompt_support(model_id):
                # system_message = (
                #     f"{self._prompt}\nAfter each response, type 'TERMINATE' to "
                #     "end the conversation"
                # )
                system_message = self._prompt
            else:
                system_message = None

            self.agent = AssistantAgent(
                name=agent,
                model_client=self.model_client,
                tools=tools,
                system_message=system_message,
                token_callback=callback,
                reflect_on_tool_use=True,
            )

            # Load Extra system messages
            if "extra_context" not in self._agents[agent]:
                self._agents[agent]["extra_context"] = []
            for extra in self._agents[agent]["extra_context"]:
                if extra[0] == "ai":
                    self.agent._model_context.add_message(
                        AssistantMessage(content=extra[1], source=agent)
                    )
                elif extra[0] == "human":
                    self.agent._model_context.add_message(
                        UserMessage(content=extra[1], source="user")
                    )
                elif extra[0] == "system":
                    raise ValueError(f"system message not implemented: {extra[0]}")
                else:
                    raise ValueError(f"Unknown extra context type {extra[0]}")

            termination_1 = MaxMessageTermination(max_messages=10)
            self.terminator = ExternalTermination()
            termination = termination_1 | self.terminator
            self.agent_team = RoundRobinGroupChat(
                [self.agent], termination_condition=termination
            )

    def _create_team_round_robin(self, agent_data: dict) -> RoundRobinGroupChat:
        """Create a team of agents that will operate in a round-robin fashion

        Parameters
        ----------
        agent_data : dict
            description of the agent team

        Returns
        -------
        RoundRobinGroupChat
            autogen RoundRobinGroupChat object
        """

        # agent_data needs to be a team
        if "type" not in agent_data or agent_data["type"] != "team":
            raise ValueError("agent_data for round_robin team must be a team")

        agents = []
        for agent in agent_data["agents"]:
            subagent_data = self._agents[agent]
            if "model" in subagent_data:
                model_client = self.mm.open_model(subagent_data["model"])
            else:
                model_client = self.mm.open_model(self.mm.default_chat_model)

            # don't use tools if the model does't support them
            if (
                not self.mm.get_tool_support(subagent_data["model"])
                or not model_client.capabilities["function_calling"]
                or "tools" not in subagent_data
            ):
                tools = None
            else:
                # load the tools
                tools = []
                for tool in subagent_data["tools"]:
                    tools.append(self.tools[tool])

            agents.append(
                AssistantAgent(
                    name=agent,
                    model_client=model_client,
                    tools=tools,
                    system_message=subagent_data["description"],
                    reflect_on_tool_use=True,
                )
            )

        max_rounds = agent_data["max_rounds"] if "max_rounds" in agent_data else 5

        text_termination = TextMentionTermination(agent_data["termination_message"])
        max_message_termination = MaxMessageTermination(max_rounds)
        termination = text_termination | max_message_termination
        team = RoundRobinGroupChat(agents, termination_condition=termination)
        return team

    async def ask(self, message: str) -> TaskResult:
        async def field_responses(
            agent_run: AsyncIterable, oneshot: bool, **kwargs
        ) -> TaskResult:
            """Run the agent (or team) and handle the responses

            Parameters
            ----------
            agent_run : AsyncIterable
                Function to run the agent
            oneshot : bool
                should the agent auto terminate after the first response

            Returns
            -------
            TaskResult
            """
            async for response in agent_run(**kwargs):
                # Ingore the autogen tool call summary messages that follow a
                # tool call
                if isinstance(response, ToolCallSummaryMessage):
                    self.log(f"ignoring tool call summary message: {response.content}")
                    continue

                # if we're streaming, we don't need to show the TextMessage, we got the
                if isinstance(response, TextMessage):
                    # ignore "user", it is a repeat of the user message
                    if response.source == "user":
                        continue
                    # if we're streaming no need to show TextMessage, we got the
                    # tokens already
                    if not self._currently_streaming:
                        await self._message_callback(
                            response.content, agent=response.source, complete=True
                        )
                    if oneshot:
                        # cleanly terminate the conversation
                        self.terminator.set()
                        response = TaskResult(
                            messages=[response], stop_reason="oneshot"
                        )
                        break
                    else:
                        continue

                if isinstance(response, ToolCallRequestEvent):
                    tool_message = "calling tool\n"
                    for tool in response.content:
                        tool_message += (
                            f"{tool.name} with arguments:\n{tool.arguments}\n"
                        )
                    tool_message += "..."
                    await self._message_callback(tool_message, agent=response.source)
                    continue
                if isinstance(response, ToolCallExecutionEvent):
                    await self._message_callback(
                        "done", agent=response.source, complete=True
                    )
                    continue
                if isinstance(response, TaskResult):
                    return response
                if response is None:
                    continue
                await self._message_callback("\n\n<unknown>\n\n", flush=True)
                await self._message_callback(repr(response), flush=True)
            return response

        async def team_run_stream(task: str, oneshot: bool = False) -> TaskResult:
            """Run the team and handle the responses

            Parameters
            ----------
            task : str
                Task for the team to run
            oneshot : bool, optional
                Terminate after first response (single agent teams), by default False
            """
            # Generate a cancelation token
            self._cancelation_token = CancellationToken()
            response: TaskResult = await field_responses(
                self.agent_team.run_stream,
                oneshot=oneshot,
                task=message,
                cancellation_token=self._cancelation_token,
            )
            return response

        # Not using this, keeping it here in case I want to revisit
        async def assistant_run() -> None:
            response = await self.agent.on_messages(
                [TextMessage(content=message, source="user")],
                cancellation_token=CancellationToken(),
            )
            return response

        # Not using this as it was cumbersome and cleaner to support single-agent
        # teams, leaving it here in case I want to revisit.
        async def agent_run_stream(task: str) -> None:
            # Generate a cancelation token
            self._cancelation_token = CancellationToken()
            await field_responses(
                self.agent.on_messages_stream,
                messages=[TextMessage(content=task, source="user")],
                cancellation_token=self._cancelation_token,
            )

        # if there is only one agent in the team, just run the agent
        if len(self.agent_team._participants) == 1:
            result = await team_run_stream(task=message, oneshot=True)
        else:
            result = await team_run_stream(task=message)

        # result = await team_run_stream(task=message)
        # self.logger(f"result: {result.stop_reason}")
        self.log(f"result: {result.stop_reason}")


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
