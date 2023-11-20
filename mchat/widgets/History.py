from textual.widget import Widget
from textual.widgets import Static
from textual.widgets import Button
from textual.reactive import Reactive
from textual.message import Message
from textual.reactive import reactive
from textual.containers import VerticalScroll
from textual.pad import HorizontalPad
from textual import events
from textual.events import Click
from textual import on
from dataclasses import dataclass
from rich.console import RenderableType
from rich.text import Text, TextType
from rich.text import Text

from langchain import PromptTemplate
from langchain import LLMChain
from langchain.chat_models.base import BaseChatModel

from typing import Callable
from rich.padding import Padding
from mchat.Conversation import ConversationRecord, Turn


""" 
Design: 
History Widget is a vertical scroll container that displays a list of Chat
History sesisons represented by Clickable Static Widgets. 

"""


class HistorySessionBox(Static, can_focus=True):
    """A Static Widget that displays a single Chat History session."""

    """The text label that appears within the button."""
    label: reactive[TextType] = reactive[TextType]("")

    def __init__(
        self,
        record: ConversationRecord | None = None,
        new_label: str = "",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.record = record
        self.current = True
        if self.record is None or len(self.record.turns) == 0:
            self.label = Text(new_label)
        else:
            self.label = Text(record.turns[-1].prompt)

    def update(self, record: ConversationRecord, summary: str = "") -> None:
        """Update the HistorySessionBox with a new ConversationRecord."""

        self.record = record
        if len(summary) > 0:
            self.label = Text(summary)
            return

        if len(record.turns) == 0:
            return
        self.label = Text(record.turns[-1].prompt)

    def render(self) -> RenderableType:
        assert isinstance(self.label, Text)
        label = self.label.copy()
        label.stylize(self.rich_style)
        return HorizontalPad(
            label,
            1,
            1,
            self.rich_style,
            self._get_rich_justify() or "left",
        )

    @dataclass
    class Clicked(Message):
        clicked_box: Widget

    def _on_click(self, event: Click) -> None:
        event.stop()
        self.post_message(self.Clicked(self))


class HistoryContainer(VerticalScroll):
    """A vertical scroll container that displays a list of Chat History sessions."""

    prompt_template = """
    Here is a chat conversation in the form of 'user: message' and
    'bot: response' pairs. You will describe the conversation in 66 characters or
    less in a way that describes the essence of the conversation, not the acutal 
    conversation details: {conversation}
    """

    def __init__(
        self, summary_model: BaseChatModel, new_label: str = "", *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.new_label = new_label

        self.prompt = PromptTemplate(
            template=self.prompt_template, input_variables=["conversation"]
        )
        self.llm_chain = LLMChain(prompt=self.prompt, llm=summary_model)

    def compose(self) -> None:
        self.current = HistorySessionBox(new_label=self.new_label)
        self.current.add_class("-active")
        yield self.current

    async def _add_session(self, record: ConversationRecord | None = None) -> None:
        """Add a new session to the HistoryContainer."""
        self.current = HistorySessionBox(record=record, new_label=self.new_label)
        self.current.add_class("-active")
        # remove the .-active class from all other boxes
        for child in self.children:
            child.remove_class("-active")
        self.mount(self.current)

    async def update(self, record: ConversationRecord):
        # Format the last 10 turns of the conversation into a single string
        # to be summarized.  If there are fewer than five turns, summarize
        # the whole conversation.
        turns = record.turns
        if len(turns) > 10:
            turns = turns[-10:]
        conversation = "\n".join(
            [f"user:{turn.prompt}, bot:{turn.response}" for turn in turns]
        )
        # Summarize the conversation
        summary = self.llm_chain.run(conversation)

        self.current.update(record, summary)

    async def new_session(self) -> None:
        await self._add_session()

    @on(HistorySessionBox.Clicked)
    def history_session_box_clicked(self, event: events) -> None:
        # set the clicked box to current and set the .-active class
        self.current = event.clicked_box
        self.current.add_class("-active")
        # remove the .-active class from all other boxes
        for child in self.children:
            if child != self.current:
                child.remove_class("-active")
        self.post_message(HistoryContainer.HistorySessionClicked(self.current.record))

    class HistorySessionClicked(Message):
        def __init__(self, record: ConversationRecord) -> None:
            self.record = record
            super().__init__()
