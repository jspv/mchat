from dataclasses import dataclass
from typing import Any
from webbrowser import open_new_tab

from textual import events, on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Markdown as MarkdownWidget
from textual.widgets._markdown import MarkdownFence

"""
Design:

ChatTurn is a widget that displays a 'turn' in a chat window. Each message
back and forth will be a different ChatTurn widget. A sequence of ChatTurns
is generally displayed in a Vertical container.

Each turn will be a prompt from the user and some number of responses from
the underlying agent(s); multiple responses are possible if the agent is a
group of agents.

"""


# Testing using MarkdownWidget instead of a Static rendering Markdown
class ChatTurn(Widget):
    def __init__(self, message="", role=None, title=None, **kwargs) -> None:
        self.message = message
        self.role = role
        self.title = title
        super().__init__(classes=role, **kwargs)

    def compose(self) -> ComposeResult:
        with Vertical(classes=self.role, id="chatturn-container"):
            self.md = MarkdownWidget(classes=self.role, id="chatturn-markdown")
            if self.title:
                self.md.border_title = self.title
            yield self.md

    def on_click(self, event: events.Click) -> None:
        widget = event.widget
        if widget is None:
            return
        local_text = (
            widget.parent.code if isinstance(widget.parent, MarkdownFence) else None
        )
        self.post_message(ChatTurn.ChatTurnClicked(widget=self, local_text=local_text))

    @property
    def markdown(self):
        return self.message

    @on(MarkdownWidget.LinkClicked)
    def on_link_clicked(self, message: MarkdownWidget.LinkClicked) -> None:
        """Open the clicked link in a new tab."""
        # HACK Fixing the URL to be clickable - reported textual bug #5426
        # https://github.com/Textualize/textual/issues/5426#issuecomment-2558582499
        from urllib.parse import quote, urlparse

        parsed_url = urlparse(message.href)
        query = parsed_url.query
        query_quoted = quote(query, safe="/=&")
        new_http = (
            f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}?{query_quoted}"
        )

        self.log.debug(f"Link clicked: \n{quote(message.href, safe=':/')}\n\n")
        self.log.debug(f"Actual message: \n{self.message}\n\n]")
        self.log.debug(f"Fixed? \n{new_http}\n\n")
        open_new_tab(new_http)

    async def append_chunk(self, chunk: Any):
        self.message += chunk
        await self.md.update(self.message)

    # textual message to be sent when clicked
    @dataclass
    class ChatTurnClicked(Message):
        widget: Widget
        local_text: str | None


# class OldChatTurn(Widget, can_focus=True):
#     def __init__(self, message="", role=None, **kwargs) -> None:
#         self.message = message
#         self.role = role
#         super().__init__(classes=role, **kwargs)

#     def compose(self) -> ComposeResult:
#         with Vertical(classes=self.role, id="chatturn-container"):
#             self.md = Static(classes=self.role, id="chatturn-markdown")
#             yield self.md

#     def on_mount(self) -> None:
#         pass

#     def on_click(self, event: events.Click) -> None:
#         self.log.debug("Clicked on ChatTurn")
#         self.post_message(ChatTurn.ChatTurnClicked(self))

#     def get_content_width(self, container: Size, viewport: Size) -> int:
#         # Naive approach. Can sometimes look strange, but works well enough.
#         return min(len(self.message), container.width)

#     @property
#     def markdown(self) -> Markdown:
#         return Markdown(self.message)

#     # def render(self) -> RenderableType:
#     #     self.md.update(self.markdown)
#     #     # return self.markdown

#     async def append_chunk(self, chunk: Any):
#         self.message += chunk
#         self.md.update(self.markdown)
#         # self.refresh(layout=True)

#     # textual message to be sent when clicked
#     @dataclass
#     class ChatTurnClicked(Message):
#         widget: Widget
#         """ The widget that was clicked."""
