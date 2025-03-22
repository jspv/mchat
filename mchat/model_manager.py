import asyncio
import copy
import logging
from dataclasses import dataclass, fields
from typing import Literal

from autogen_ext.models.openai import (
    AzureOpenAIChatCompletionClient,
    OpenAIChatCompletionClient,
)
from dynaconf import DynaconfFormatError
from openai import OpenAI
from pydantic.networks import HttpUrl

from config import settings

from .azure_auth import AzureADTokenProvider

logger = logging.getLogger(__name__)


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
    model_info: dict | None = None
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


class DallEAPIWrapper:
    def __init__(
        self,
        api_key: str | None = None,
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
        self.client = OpenAI(api_key=self.api_key)

    def run(self, query: str) -> str:
        try:
            response = self.client.images.generate(
                prompt=query,
                n=self.num_images,
                size=self.size,
                model=self.model,
                quality=self.quality,
                response_format="url",
            )
            image_urls = self.separator.join([item.url for item in response.data])
            return image_urls if image_urls else "No image was generated"
        except Exception as e:
            return f"Image Generation Error: {str(e)}"

    async def arun(self, query: str) -> str:
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
            return f"Image Generation Error: {str(e)}"


class ModelManager:
    def __init__(self):
        try:
            self.model_configs = settings["models"].to_dict()
            self.config: dict[str, ModelConfig] = {}
        except AttributeError as e:
            missing_attr = str(e).split("'")[-2]
            error_message = (
                f"Missing required setting: '{missing_attr}'. "
                "Please add it to your settings file."
            )
            raise RuntimeError(error_message) from e
        except DynaconfFormatError as e:
            if "has no attribute" in str(e):
                missing_attr = str(e).split("'")[-2]
                error_message = (
                    f"Missing required setting: '{missing_attr}'. "
                    "Please add it to your settings or secrets file."
                )
                raise RuntimeError(error_message) from e
            else:
                raise RuntimeError(f"Dynaconf encountered an error: {e}") from e

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

        for model_type in self.model_configs.keys():
            for model_id, model_config in self.model_configs[model_type].items():
                self.config[model_id] = model_types[model_type][
                    model_config["api_type"]
                ](model_id=model_id, model_type=model_type, **model_config)

        self.azure_token_provider = None
        for azure_model in self.filter_models({"api_type": ["azure"]}):
            if self.config[azure_model].api_key == "provider":
                self.azure_token_provider = AzureADTokenProvider()
                break

        self.default_chat_model = settings.defaults.chat_model
        self.default_image_model = settings.get("defaults.image_model", None)
        self.default_embedding_model = settings.get("defaults.embedding_model", None)
        self.default_chat_temperature = settings.defaults.chat_temperature
        self.default_memory_model = settings.defaults.memory_model
        self.default_memory_model_temperature = (
            settings.defaults.memory_model_temperature
        )
        self.default_memory_model_max_tokens = settings.defaults.memory_model_max_tokens

    def filter_models(self, filter_dict):
        return [
            key
            for key, value in self.config.items()
            if all(getattr(value, k) in v for k, v in filter_dict.items())
        ]

    def open_model(self, model_id: str, **kwargs) -> object:
        logger.debug(f"Opening model {model_id}")
        record = copy.deepcopy(self.config[model_id])
        model_kwargs = {
            **{
                field.name: getattr(record, field.name)
                for field in fields(record)
                if not field.name.startswith("_")
            },
            **kwargs,
        }
        model_kwargs.pop("api_type")
        model_kwargs.pop("model_type")

        if record.api_type == "azure" and model_kwargs["api_key"] == "provider":
            model_kwargs["azure_ad_token_provider"] = self.azure_token_provider
            model_kwargs.pop("api_key")

        if record.model_type == "chat":
            if not getattr(record, "_temperature_support", True):
                model_kwargs.pop("temperature")
            elif getattr(record, "temperature", None) is None:
                model_kwargs.setdefault("temperature", self.default_chat_temperature)

            if record.api_type == "open_ai":
                return OpenAIChatCompletionClient(**model_kwargs)
            elif record.api_type == "azure":
                return AzureOpenAIChatCompletionClient(**model_kwargs)
        elif record.model_type == "image":
            return DallEAPIWrapper(**model_kwargs)

        raise ValueError("Invalid model_type")

    def get_streaming_support(self, model_id: str) -> bool:
        return self.config[model_id]._streaming_support

    def get_tool_support(self, model_id: str) -> bool:
        return self.config[model_id]._tool_support

    def get_system_prompt_support(self, model_id: str) -> bool:
        return self.config[model_id]._system_prompt_support

    def get_temperature_support(self, model_id: str) -> bool:
        return self.config[model_id]._temperature_support

    def get_structured_output_support(self, model_id: str) -> bool:
        return self.config[model_id]._structured_output_support

    def get_compatible_models(self, agent: str, agents: dict) -> list:
        filter = {"model_type": ["chat"]}
        if "tools" in agents[agent]:
            filter["_tool_support"] = [True]
        return self.filter_models(filter)

    @property
    def available_image_models(self) -> list:
        return self.filter_models({"model_type": ["image"]})

    @property
    def available_chat_models(self) -> list:
        return self.filter_models({"model_type": ["chat"]})

    @property
    def available_embedding_models(self) -> list:
        return self.filter_models({"model_type": ["embedding"]})
