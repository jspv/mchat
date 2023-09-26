from typing import Any

from rich.console import RenderableType

from rich.markdown import Markdown

from textual.widgets import Markdown as MarkdownWidget, Static
from textual.containers import Vertical
from textual.widget import Widget

from textual.geometry import Size


# Testing using MarkdownWidget instead of a Static rendering Markdown
class newChatTurn(Widget):
    def compose(self) -> None:
        with Vertical(classes=self.role, id="chatturn-container"):
            self.md = MarkdownWidget(classes=self.role, id="chatturn-markdown")
            yield self.md

    def __init__(self, message="", role=None, *args, **kwargs) -> None:
        self.message = message
        self.role = role
        super().__init__(classes=role, *args, **kwargs)

    async def append_chunk(self, chunk: Any):
        self.app.log.debug(f'Appending chunk: "{chunk}"')
        self.message += chunk
        self.app.log.debug(f"New message: {self.message}")
        self.md.update(self.message)

    @property
    def markdown(self):
        return self.message


class ChatTurn(Widget, can_focus=True):
    def __init__(self, message="", role=None, *args, **kwargs) -> None:
        self.role = role
        self.message = message
        super().__init__(classes=role, *args, **kwargs)

    def compose(self) -> None:
        with Vertical(classes=self.role, id="chatturn-container"):
            self.md = Static(classes=self.role, id="chatturn-markdown")
            yield self.md

    def on_mount(self) -> None:
        pass

    def get_content_width(self, container: Size, viewport: Size) -> int:
        # Naive approach. Can sometimes look strange, but works well enough.
        return min(len(self.message), container.width)

    @property
    def markdown(self) -> Markdown:
        # foo = "| Simple | Table |\n| ------ | ----- "
        # self.log.debug(f"foo is {foo}")
        # return Markdown(foo)
        return Markdown(self.message)

    # def render(self) -> RenderableType:
    #     self.md.update(self.markdown)
    #     # return self.markdown

    async def append_chunk(self, chunk: Any):
        self.app.log.debug(f"Appending chunk: {chunk}")
        self.message += chunk
        self.app.log.debug(f"New message: {self.message}")
        self.md.update(self.markdown)
        # self.refresh(layout=True)
