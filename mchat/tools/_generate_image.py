import asyncio
from typing import Annotated, Literal

from openai import OpenAI

from config import settings
from mchat.tool_utils import BaseTool


class OpenAIImageTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="generate_image",
            description=(
                "Generates an image using OpenAI's DALL-E API. it will return "
                "a url and a revised_prompt if the tool decided to enhance the "
                "prompt. Do not alter the url in any way, and provide information "
                "back to the user if the prompt was changed"
            ),
        )

    def verify_setup(self):
        api_key = settings.get("openai_api_key", None)
        if not api_key:
            raise ValueError(f"{self.name} OpenAI API key not found.")
        self.wrapper = OpenAIImageAPIWrapper(api_key=api_key)

    def run(
        self,
        prompt: Annotated[str, "Prompt for image generation."],
        model: Annotated[Literal["dall-e-2", "dall-e-3"], "Model to use."] = "dall-e-2",
        num_images: Annotated[int, "Number of images to generate."] = 1,
        size: Annotated[
            Literal["1024x1024", "1024x1792", "1792x1024", "256x256", "512x512"],
            (
                "Size of images. dall-e-2 supports 1024x1024, 256x256, 512x512, "
                "dall-e-3 supports 1024x1024, 1024x1792, 1792x1024"
            ),
        ] = "1024x1024",
        quality: Annotated[Literal["standard", "hd"], "Quality of images"] = "standard",
    ) -> dict:
        """
        Generate an image using OpenAI's DALL-E API.

        Args:
            prompt (str): Prompt for image generation.
            model (str): Model to use (dall-e-2 or dall-e-3).
            num_images (int): Number of images to generate.
            size (str): Size of images.
            quality (str): Quality of images.

        Returns:
            dict: Image data containing the revised prompt and URL.
        """
        if not self.is_callable:
            raise RuntimeError(
                f"Tool '{self.name}' is not callable due to setup failure: {self.load_error}"
            )

        image_data = self.wrapper.generate_image(
            prompt=prompt,
            model=model,
            num_images=num_images,
            size=size,
            quality=quality,
        )
        return image_data


class OpenAIImageAPIWrapper:
    """Wrapper for OpenAI's DALL-E Image Generator."""

    def __init__(self, api_key: str, log_file: str | None = None):
        self.api_key = api_key

        self.log_file = (
            settings.get("generate_image_log_file", None)
            if log_file is None
            else log_file
        )

        # Set the API key for OpenAI
        self.client = OpenAI(api_key=self.api_key)

    # below is the async version of the function, not yet integrated
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

    def generate_image(
        self,
        prompt: Annotated[str, "Prompt for image generation."],
        model: Annotated[Literal["dall-e-2", "dall-e-3"], "Model to use."] = "dall-e-2",
        num_images: Annotated[int, "Number of images to generate."] = 1,
        size: Annotated[
            Literal["1024x1024", "1024x1792", "1792x1024", "256x256", "512x512"],
            (
                "Size of images. dall-e-2 supports 1024x1024, 256x256, 512x512, "
                "dall-e-3 supports 1024x1024, 1024x1792, 1792x1024"
            ),
        ] = "1024x1024",
        quality: Annotated[Literal["standard", "hd"], "Quality of images"] = "standard",
    ):
        """Generate an image using OpenAI's DALL-E API."""
        response_format = "url"

        kwargs = {
            "prompt": prompt,
            "model": model,
            "n": num_images,
            "size": size,
            "quality": quality,
            "response_format": response_format,
        }

        try:
            response = self.client.images.generate(**kwargs)

            if self.log_file:
                with open(self.log_file, "a") as f:
                    f.write(f"Request: {repr(kwargs)}\n")
                    f.write(f"Response: {repr(response)}\n\n")
            image_data = [
                {"revised_prompt": image.revised_prompt, "url": image.url}
                for image in response.data
            ]
            return image_data if image_data else "No image was generated"
        except Exception as e:
            return f"Image Generatiom Error: {str(e)}"
