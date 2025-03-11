import asyncio
import copy
import importlib
import logging
import os
from dataclasses import dataclass, fields
from functools import reduce
from typing import (
    AsyncIterable,
    Callable,
    Dict,
    Literal,
    Optional,
)

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import (
    ExternalTermination,
    MaxMessageTermination,
    TextMentionTermination,
)
from autogen_agentchat.messages import (
    ModelClientStreamingChunkEvent,
    MultiModalMessage,
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
)
from autogen_agentchat.teams import (
    MagenticOneGroupChat,
    RoundRobinGroupChat,
    SelectorGroupChat,
)
from autogen_core import CancellationToken
from autogen_core.models import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import FunctionTool
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.models.openai import (
    AzureOpenAIChatCompletionClient,
    OpenAIChatCompletionClient,
)
from dynaconf import DynaconfFormatError
from openai import OpenAI
from pydantic.networks import HttpUrl

from config import settings

from .azure_auth import AzureADTokenProvider
from .tool_utils import BaseTool

requests_log = logging.getLogger("requests.packages.urllib3")
# requests_log.setLevel(logging.DEBUG)
requests_log.setLevel(logging.WARNING)
requests_log.propagate = True
# http.client.HTTPConnection.debuglevel = 1

logger = logging.getLogger(__name__)

intent_prompt = (
    "A human is having a conversation with an AI, you are watching that conversation"
    " and your job is to determine what the human is intending to do in their last"
    " message to the AI.  Reply with one of the following options:  1) if the human is"
    " asking to change their agent, reply with the word 'agent', 2) if the human is"
    " asking a new question unrelated to the current conversation, reply with the word"
    " 'ask', 3) if the human is continuing an existing conversation, reply with the"
    " word 'continue' 4) if the human is responding to a question that the AI asked,"
    " reply with the word 'reply', 5) if the human is trying to open or loading a file,"
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
    _structured_output_support: bool | None = None
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
    _structured_output_support: bool | None = None
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
    _cost_output: float | None = None
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
    def __init__(self):
        # Load all the models from the settings

        try:
            # Attempt to access settings
            # self.openai_api_key = settings.openai_api_key
            self.model_configs = settings["models"].to_dict()
            self.config: Dict[str, ModelConfig] = {}

        except AttributeError as e:
            # Directly missing attribute within Dynaconf
            missing_attr = str(e).split("'")[-2]
            error_message = (
                f"AMissing required setting: '{missing_attr}'. Please "
                f"add it to your settings file."
            )
            raise RuntimeError(error_message) from e

        except DynaconfFormatError as e:
            # Handle cases where Dynaconf fails due to interpolation errors
            if "has no attribute" in str(e):
                missing_attr = str(e).split("'")[-2]
                error_message = (
                    f"Missing required setting: '{missing_attr}'. Please "
                    f"add it to your settings or secrets file OR remove the "
                    f"models using it from the settings file."
                )
                raise RuntimeError(error_message) from e
            else:
                # If it's another Dynaconf error, log and raise it
                raise RuntimeError(f"Dynaconf encountered an error: {e}") from e

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
        # result is a dictionary of model_id to model_config with model_config
        # being the appropriate dataclass
        for model_type in self.model_configs.keys():
            for model_id, model_config in self.model_configs[model_type].items():
                self.config[model_id] = model_types[model_type][
                    model_config["api_type"]
                ](model_id=model_id, model_type=model_type, **model_config)

        # if there are any azure models, initialize the token provider
        if self.filter_models({"api_type": ["azure"]}):
            self.azure_token_provider = AzureADTokenProvider()
        else:
            self.azure_token_provider = None

        # Set the default models
        self.default_chat_model = settings.defaults.chat_model
        self.default_image_model = settings.get("defaults.image_model", None)
        self.default_embedding_model = settings.get("defaults.embedding_model", None)
        self.default_chat_temperature = settings.defaults.chat_temperature
        self.default_memory_model = settings.defaults.memory_model
        self.default_memory_model_temperature = (
            settings.defaults.memory_model_temperature
        )
        self.default_memory_model_max_tokens = settings.defaults.memory_model_max_tokens

    @property
    def available_image_models(self) -> list:
        """Returns a list of available image models"""
        return self.filter_models({"model_type": ["image"]})

    @property
    def available_chat_models(self) -> list:
        """Returns a list of available chat models"""
        return self.filter_models({"model_type": ["chat"]})

    @property
    def available_embedding_models(self) -> list:
        """Returns a list of available embedding models"""
        return self.filter_models({"model_type": ["embedding"]})

    def filter_models(self, filter_dict):
        """Filter the config list by provider and model.

        Args:
            filter_dict (dict, optional): The filter dict with keys corresponding to a
            field in each config, and values corresponding to lists of acceptable values
            for each key.

        Example: filter_dict =
        {"model_type": ["chat"], "_streaming_support":[True]}

        Returns:
            list: The filtered config list.
        """
        return [
            key
            for key, value in self.config.items()
            if all(getattr(value, k) in v for k, v in filter_dict.items())
        ]

    def get_compatible_models(self, agent: str, agents: dict) -> list:
        """Returns a list of models that are compatible with the agent"""

        filter = {"model_type": ["chat"]}

        # if agent doesn't specify a chat model, allow any chat model
        # model = agents[agent].get("model", self.default_chat_model)

        # check if the agent needs tools
        if "tools" in agents[agent]:
            filter["_tool_support"] = [True]

        # check if the agent needs system messages
        if "prompt" in agents[agent] and agents[agent]["prompt"] != "":
            filter["_system_prompt_support"] = [True]

        return self.filter_models(filter)

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

    def get_structured_output_support(self, model_id: str) -> bool:
        """Returns whether the model supports structured output, assume true"""
        return self.config[model_id]._structured_output_support

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
                    # api_key=model_kwargs["api_key"],
                    azure_ad_token_provider=self.azure_token_provider,
                    api_version=model_kwargs["api_version"],
                    azure_endpoint=model_kwargs["azure_endpoint"],
                )
            return model

        elif record.model_type == "image":
            model = DallEAPIWrapper(**model_kwargs)
            return model
        else:
            raise ValueError("Invalid model_type")


class DallEAPIWrapper:
    """Wrapper for OpenAI's DALL-E Image Generator."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        num_images: int = 1,
        size: str = "1024x1024",
        separator: str = "\n",
        model: str | None = "dall-e-2",
        quality: str | None = "standard",
        **kwargs,
    ):
        self.api_key = api_key
        self.num_images = num_images
        self.size = size
        self.separator = separator
        self.model = model
        self.quality = quality

        # Set the API key for OpenAI
        self.client = OpenAI(api_key=self.api_key)

    def run(self, query: str) -> str:
        """Run query through OpenAI and parse result."""
        try:
            response = self.client.images.generate(
                n=self.num_images,
                size=self.size,
                model=self.model,
                quality=self.quality,
                response_format="url",
            )
            image_urls = self.separator.join([item["url"] for item in response["data"]])
            return image_urls if image_urls else "No image was generated"
        except Exception as e:
            return f"Image Generation Error: {str(e)}"

    async def arun(self, query: str) -> str:
        """Run query through OpenAI and parse result asynchronously."""
        # loop = asyncio.get_running_loop()
        try:
            response = await asyncio.to_thread(
                lambda: self.client.images.generate(
                    prompt=query,
                    n=self.num_images,
                    size=self.size,
                    model=self.model,
                    quality=self.quality,
                    response_format="url",
                )
            )
            image_urls = self.separator.join([item.url for item in response.data])
            return image_urls if image_urls else "No image was generated"
        except Exception as e:
            return f"Image Generatiom Error: {str(e)}"


class AutogenManager(object):
    def __init__(
        self,
        message_callback: Callable | None = None,
        agents: dict | None = None,
        stream_tokens: bool = True,
    ):
        self.mm = ModelManager()
        self._message_callback = message_callback  # send messages back to the UI
        self._agents = agents if agents is not None else {}
        self._stream_tokens = stream_tokens  # streaming currently enabled or not
        self._streaming_preference = stream_tokens  # remember the last setting

        # model being used by the agent, will be set when a new conversation is started
        self._model_id = None

        # Will be used to cancel ongoing tasks
        self._cancelation_token = None

        # Inialize available tools
        self.tools = {}
        # Get the directory of the current file
        current_directory = os.path.dirname(os.path.abspath(__file__))
        tools_directory = os.path.join(current_directory, "tools")

        for filename in os.listdir(tools_directory):
            if filename.endswith(".py"):
                file_path = os.path.join(tools_directory, filename)
                spec = importlib.util.spec_from_file_location(filename[:-3], file_path)
                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)

                    # Assuming all tool classes are derived from BaseTool
                    for item_name in dir(module):
                        item = getattr(module, item_name)
                        if (
                            isinstance(item, type)
                            and issubclass(item, BaseTool)
                            and item is not BaseTool
                        ):
                            tool_instance = item()  # Instantiate the tool
                            if tool_instance.is_callable:
                                self.tools[tool_instance.name] = FunctionTool(
                                    tool_instance.run,
                                    description=tool_instance.description,
                                    name=tool_instance.name,
                                )
                            else:
                                logger.warning(
                                    f"Tool {tool_instance.name} not loaded due to "
                                    f"setup failure: {tool_instance.load_error}"
                                )
                except Exception as e:
                    logger.warning(f"Failed to load tool module {filename[:-3]}: {e}")

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

    @property
    def description(self) -> str:
        """Returns the current prompt"""
        # return self.agent.prompt
        return self._description

    async def get_memory(self) -> str:
        """Returns the current memory in a recoverable text format"""
        # save_state is async, so we need to await it
        return await self.agent.save_state()

    @property
    def stream_tokens(self) -> bool | None:
        """Are we currently streaming tokens if they are supported

        Returns:
        -------
        bool | None
            True if currently streaming tokens, False if not, None if not supported
        """

        return self._stream_tokens

    @stream_tokens.setter
    def stream_tokens(self, value: bool) -> None:
        """enable or disable token streaming"""

        if not isinstance(value, bool):
            raise ValueError("stream_tokens must be a boolean")

        # If currently disabled by logic, don't allow it to be enabled
        if self._stream_tokens is None:
            logger.info(
                f"token streaming disabled, setting stream_tokens to {value} ignored"
            )
            return

        # This remembers the last setting, so that if the model doesn't support
        # streaming, we know what to return to when switching to one that does
        self._streaming_preference = value
        self._stream_tokens = value
        self._set_agent_streaming()

    def _set_agent_streaming(self) -> None:
        """Reset the streaming callback for the agent"""

        value = self._stream_tokens

        if hasattr(self, "agent"):
            # HACK TODO - this needs to becone a public toggle
            self.agent._model_client_stream = value
            logger.info(f"token streaming for {self.agent.name} set to {value}")
        else:
            raise ValueError("stream_tokens can only be set if there is an agent")

    @property
    def model(self) -> str:
        """Returns the current model"""
        return self._model_id

    def cancel(self) -> None:
        """Cancel the current conversation"""
        if self._cancelation_token:
            self._cancelation_token.cancel()

    def terminate(self) -> None:
        """Terminate the current conversation"""
        if hasattr(self, "terminator"):
            self.terminator.set()

    def clear_memory(self) -> None:
        """Clear the memory"""
        if hasattr(self, "agent") and hasattr(self.agent, "_model_context:"):
            self.agent._model_context.clear()

    async def update_memory(self, state: dict) -> None:
        await self.agent.load_state(state)

    async def new_conversation(
        self,
        agent: str,
        model_id: str,
        temperature: float = 0.0,
        # TODO streaming should be a specified default, not hard coded
        stream_tokens: bool = True,
    ) -> None:
        """Intialize a new conversation with the given agent and model

        Parameters
        ----------
        agent : dict
            agent object
        model_id: str
            model to use""
        temperature : float, optional
            model temperature setting, by default 0.0
        stream_tokens: bool, optional
            if the model should stream tokens back to the UI
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

        self._prompt = (
            self._agents[agent]["prompt"] if "prompt" in self._agents[agent] else ""
        )

        self._description = (
            self._agents[agent]["description"]
            if "description" in self._agents[agent]
            else ""
        )

        # TODO: what to do when the model is specified for a team-based agent that
        # currently this is ingored.
        if "type" in agent_data and agent_data["type"] == "team":
            # Team-based Agents
            self.agent_team = self._create_team(agent_data["team_type"], agent_data)
            self.agent = self.agent_team._participants[0]

            # currently not streaming tokens for team agents
            self._stream_tokens = None
            logger.info("token streaming for team-based agents currently disabled")

        else:
            # Solo Agent
            self.model_client = self.mm.open_model(model_id)
            self._model_id = model_id

            # don't use tools if the model does't support them
            if (
                not self.mm.get_tool_support(model_id)
                or not self.model_client.model_info["function_calling"]
                or "tools" not in agent_data
            ):
                tools = None
            else:
                tools = [
                    self.tools[tool]
                    for tool in agent_data["tools"]
                    if tool in self.tools
                ]

            # Don't use system messages if not supported
            if self.mm.get_system_prompt_support(model_id):
                system_message = self._prompt
            else:
                system_message = None

            # build the agent
            if "type" in agent_data and agent_data["type"] == "autogen-agent":
                if agent_data["name"] == "websurfer":
                    self.agent = MultimodalWebSurfer(
                        model_client=self.model_client,
                        name=agent,
                    )
                    # not streaming builtin autogen agents right now
                    logger.info(
                        f"token streaming agent:{agent} disabled or not supported"
                    )
                    self._stream_tokens = None

                else:
                    raise ValueError(f"Unknown autogen agent type for agent:{agent}")
            else:
                self.agent = AssistantAgent(
                    name=agent,
                    model_client=self.model_client,
                    tools=tools,
                    system_message=system_message,
                    model_client_stream=True,
                    reflect_on_tool_use=True,
                )

            # Load Extra system messages if they exist
            if "extra_context" not in self._agents[agent]:
                self._agents[agent]["extra_context"] = []
            for extra in self._agents[agent]["extra_context"]:
                if extra[0] == "ai":
                    await self.agent._model_context.add_message(
                        AssistantMessage(content=extra[1], source=agent)
                    )
                elif extra[0] == "human":
                    await self.agent._model_context.add_message(
                        UserMessage(content=extra[1], source="user")
                    )
                elif extra[0] == "system":
                    raise ValueError(f"system message not implemented: {extra[0]}")
                else:
                    raise ValueError(f"Unknown extra context type {extra[0]}")

            # Set streaming to the current preference (if supported)

            # disable streaming if not supported by the model, otherwse use preference
            if not self.mm.get_streaming_support(model_id):
                self._stream_tokens = None
                self._set_agent_streaming()
                logger.info(f"token streaming for model {model_id} not supported")
            else:
                self._stream_tokens = self._streaming_preference
                self._set_agent_streaming()

            # Build the termination conditions
            terminators = []
            max_rounds = agent_data["max_rounds"] if "max_rounds" in agent_data else 5
            terminators.append(MaxMessageTermination(max_rounds))
            if "termination_message" in agent_data:
                terminators.append(
                    TextMentionTermination(agent_data["termination_message"])
                )
            self.terminator = ExternalTermination()  # for custom terminations
            terminators.append(self.terminator)
            termination = reduce(lambda x, y: x | y, terminators)

            # if there is only one agent, set oneshot to true
            self.oneshot = (
                True if "oneshot" not in agent_data else agent_data["oneshot"]
            )

            self.agent_team = RoundRobinGroupChat(
                [self.agent], termination_condition=termination
            )

    def _create_team(
        self, team_type: str, agent_data: dict
    ) -> RoundRobinGroupChat | SelectorGroupChat | MagenticOneGroupChat:
        """Create a team of agents

        Parameters
        ----------
        team_type : str
            type of team to create
        agent_data : dict
            description of the agent team
        """

        # agent_data needs to be a team
        if "type" not in agent_data or agent_data["type"] != "team":
            raise ValueError("agent_data 'type' for team must be 'team'")

        # build the agents
        agents = []
        for agent in agent_data["agents"]:
            subagent_data = self._agents[agent]
            if "model" not in subagent_data:
                subagent_data["model"] = self.mm.default_chat_model
            model_client = self.mm.open_model(subagent_data["model"])

            if "type" in subagent_data and subagent_data["type"] == "autogen-agent":
                if subagent_data["name"] == "websurfer":
                    agents.append(
                        MultimodalWebSurfer(
                            model_client=model_client,
                            name=agent,
                        )
                    )
                else:
                    raise ValueError(f"Unknown autogen agent type for agent:{agent}")
            else:
                # don't use tools if the model does't support them
                if (
                    not self.mm.get_tool_support(subagent_data["model"])
                    or not model_client.model_info["function_calling"]
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
                        system_message=subagent_data["prompt"],
                        description=subagent_data["description"],
                        reflect_on_tool_use=True,
                    )
                )

        # constuct the team
        terminators = []
        max_rounds = agent_data["max_rounds"] if "max_rounds" in agent_data else 5
        terminators.append(MaxMessageTermination(max_rounds))
        if "termination_message" in agent_data:
            terminators.append(
                TextMentionTermination(agent_data["termination_message"])
            )
        self.terminator = ExternalTermination()  # for custom terminations
        terminators.append(self.terminator)
        termination = reduce(lambda x, y: x | y, terminators)

        if "oneshot" in agent_data:
            self.oneshot = agent_data["oneshot"]
        else:
            self.oneshot = True if len(agents) == 1 else False

        if team_type == "round_robin":
            return RoundRobinGroupChat(agents, termination_condition=termination)
        elif team_type == "selector":
            if "team_model" not in agent_data:
                team_model = self.mm.open_model(self.mm.default_chat_model)
            else:
                team_model = self.mm.open_model(agent_data["team_model"])
            allow_repeated_speaker = agent_data.get("allow_repeated_speaker", False)

            if "selector_prompt" in agent_data:
                return SelectorGroupChat(
                    agents,
                    model_client=team_model,
                    selector_prompt=agent_data["selector_prompt"],
                    termination_condition=termination,
                    allow_repeated_speaker=allow_repeated_speaker,
                )
            else:
                return SelectorGroupChat(
                    agents,
                    model_client=team_model,
                    allow_repeated_speaker=allow_repeated_speaker,
                    termination_condition=termination,
                )
        elif team_type == "magnetic_one":
            if "team_model" not in agent_data:
                team_model = self.mm.open_model(self.mm.default_chat_model)
            else:
                team_model = self.mm.open_model(agent_data["team_model"])
            return MagenticOneGroupChat(
                agents, model_client=team_model, termination_condition=termination
            )
        else:
            raise ValueError(f"Unknown team type {team_type}")

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
                    logger.debug(
                        f"ignoring tool call summary message: {response.content}"
                    )
                    continue

                if isinstance(response, ModelClientStreamingChunkEvent):
                    await self._message_callback(
                        response.content, agent=response.source, complete=False
                    )
                    continue

                if isinstance(response, TextMessage):
                    # ignore "user", it is a repeat of the user message
                    if response.source == "user":
                        continue
                    # if we're streaming no need to show TextMessage, we got the
                    # tokens already
                    logger.info(f"TextMessage: {response.content}")
                    if not self._stream_tokens:
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

                if isinstance(response, MultiModalMessage):
                    await self._message_callback(
                        f"MM:{response.content}", agent=response.source, complete=True
                    )
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
                    logger.info(f"tool call result: {response.content}")
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

        self._cancelation_token = CancellationToken()
        result: TaskResult = await field_responses(
            self.agent_team.run_stream,
            oneshot=self.oneshot,
            task=message,
            cancellation_token=self._cancelation_token,
        )
        self._cancelation_token = None
        logger.info(f"result: {result.stop_reason}")


class LLMTools:
    def __init__(self):
        """Builds the intent prompt template"""

        self.mm = ModelManager()

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
