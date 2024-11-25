from typing import Any

from dataclasses import dataclass

from rich.markdown import Markdown

from textual.widgets import Markdown as MarkdownWidget, Static
from textual.containers import Vertical
from textual.widget import Widget
from textual.message import Message
from textual import on

from textual.geometry import Size

from webbrowser import open_new_tab


"""
Design:

ChatTurn is a widget that displays a 'turn' in a chat window. Each message
back and forth will be a different ChatTurn widget. A sequence of ChatTurns 
is generally displayed in a Vertical container.

"""


# Testing using MarkdownWidget instead of a Static rendering Markdown
class ChatTurn(Widget):
    def __init__(self, message="", role=None, *args, **kwargs) -> None:
        self.message = message
        self.role = role
        super().__init__(classes=role, *args, **kwargs)

    def compose(self) -> None:
        with Vertical(classes=self.role, id="chatturn-container"):
            self.md = MarkdownWidget(classes=self.role, id="chatturn-markdown")
            yield self.md

    def on_click(self) -> None:
        self.post_message(ChatTurn.ChatTurnClicked(self))

    @property
    def markdown(self):
        return self.message

    @on(MarkdownWidget.LinkClicked)
    def on_link_clicked(self, message: MarkdownWidget.LinkClicked) -> None:
        self.log.debug(f"Link clicked: {message.href}")
        open_new_tab(message.href)

    async def append_chunk(self, chunk: Any):
        self.message += chunk
        await self.md.update(self.message)

    # textual message to be sent when clicked
    @dataclass
    class ChatTurnClicked(Message):
        widget: Widget
        """ The widget that was clicked."""


class OldChatTurn(Widget, can_focus=True):
    def __init__(self, message="", role=None, *args, **kwargs) -> None:
        self.message = message
        self.role = role
        super().__init__(classes=role, *args, **kwargs)

    def compose(self) -> None:
        with Vertical(classes=self.role, id="chatturn-container"):
            self.md = Static(classes=self.role, id="chatturn-markdown")
            yield self.md

    def on_mount(self) -> None:
        pass

    def on_click(self) -> None:
        self.log.debug("Clicked on ChatTurn")
        self.post_message(ChatTurn.ChatTurnClicked(self))

    def get_content_width(self, container: Size, viewport: Size) -> int:
        # Naive approach. Can sometimes look strange, but works well enough.
        return min(len(self.message), container.width)

    @property
    def markdown(self) -> Markdown:
        return Markdown(self.message)

    # def render(self) -> RenderableType:
    #     self.md.update(self.markdown)
    #     # return self.markdown

    async def append_chunk(self, chunk: Any):
        self.message += chunk
        self.md.update(self.markdown)
        # self.refresh(layout=True)

    # textual message to be sent when clicked
    @dataclass
    class ChatTurnClicked(Message):
        widget: Widget
        """ The widget that was clicked."""
