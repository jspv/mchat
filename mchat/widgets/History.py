from textual.widget import Widget
from textual.widgets import Static
from textual.reactive import Reactive
from textual.containers import VerticalScroll
from rich.console import RenderableType
from rich.text import Text
from typing import Callable
from rich.padding import Padding
from mchat.Conversation import ConversationRecord, Turn

""" 
Design: 
History Widget is a vertical scroll container that displays a list of Chat
History sesisons represented by Clickable Static Widgets. 

"""


class History(VerticalScroll):
    """A vertical scroll container that displays a list of Chat History sessions."""

    # def compose(self) -> None:
    #     # self.mount(Static("History"))

    #     for i in range(50):
    #         yield (Static("History"))

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_session = False

    async def add_session(self, record: ConversationRecord):
        self.current = Static(record.turns[-1].prompt)
        self.mount(self.current)
        self.in_session = True

    async def update(self, record: ConversationRecord):
        if self.in_session:
            self.current.update(record.turns[-1].prompt)
        else:
            # create a new session
            await self.add_session(record)
            self.in_session = True
