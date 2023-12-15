from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI

# No Dall-e support in langhchain OpenAI, so use the native
from openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings, AzureOpenAIEmbeddings

from config import settings
from functools import partial
import asyncio
import autogen
import json
import os


class ModelManager:
    def __init__(self):
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

        self.default_chat_model = settings.defaults.chat_model
        self.default_image_model = settings.defaults.image_model
        self.default_embedding_model = settings.defaults.embedding_model
        self.default_chat_temperature = settings.defaults.chat_temperature
        self.default_memory_model = settings.defaults.memory_model
        self.default_memory_model_temperature = (
            settings.defaults.memory_model_temperature
        )
        self.default_memory_model_max_tokens = settings.defaults.memory_model_max_tokens

        os.environ["OAI_CONFIG_LIST"] = json.dumps(self.chat_config_list)

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

    def filter_config_list(self, filter_dict: dict = {}) -> list:
        """Filters config_list by filter_list"""
        return autogen.config_list_from_json(
            env_or_file="OAI_CONFIG_LIST", filter_dict=filter_dict
        )

    def open_model(
        self,
        model_id: str,
        model_type="chat",
        **kwargs,
    ) -> object:
        """Opens a model and returns the model object"""

        # Get the model record from the model_configs
        record = dict(self.model_configs[model_type][model_id])
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
                model = ChatOpenAI(**model_kwargs)
            elif record["api_type"] == "azure":
                model = AzureChatOpenAI(**model_kwargs)
            return model

        elif model_type == "image":
            model = DallEAPIWrapper(**model_kwargs)
            return model
        elif model_type == "embedding":
            model_kwargs.pop("api_type")
            if record["api_type"] == "open_ai":
                model = OpenAIEmbeddings(**model_kwargs)
            elif record["api_type"] == "azure":
                model = AzureOpenAIEmbeddings(**model_kwargs)
            return model
        else:
            raise ValueError("Invalid model_type")

    def pdf_to_vectors(
        self, pdf_file: str, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> list:
        """Converts a PDF file to a list of vectors"""
        pdf_loader = PyPDFLoader(pdf_file)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        docs = pdf_loader.load_and_split(text_splitter=text_splitter)
        print(f"Loaded {len(docs)} documents")
        embeddings = self.open_model(
            self.default_embedding_model, model_type="embedding"
        )
        vectors = Chroma.from_documents(docs, embedding=embeddings)
        return vectors


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
