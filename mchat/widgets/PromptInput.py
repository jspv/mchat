from typing import Any

from rich.console import RenderableType

from rich.markdown import Markdown

from textual.widgets import Markdown as MarkdownWidget, Input
from textual.containers import Vertical
from textual.widget import Widget

from textual.geometry import Size


class PromptInput(Input):
    pass