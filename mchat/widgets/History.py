from textual.widget import Widget
from textual.widgets import Static
from textual.widgets import Button
from textual.reactive import Reactive
from textual.message import Message
from textual.reactive import reactive
from textual.containers import VerticalScroll, Vertical, Horizontal
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


class HistorySessionBox(Widget, can_focus=True):
    """A Widget that displays a single Chat History session."""

    def __init__(
        self,
        record: ConversationRecord | None = None,
        new_label: str = "",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.new_label = new_label
        self.record = record

    def compose(self) -> None:
        with Vertical():
            with Horizontal():
                yield Static(
                    Text("..."),
                    id="history-session-box-label",
                    classes="history-session-box-top",
                )
                yield Static("", id="history-session-box-spacer")
                self.copy_button = CopyButton(id="history-session-box-copy")
                yield self.copy_button
                yield DeleteButton(id="history-session-box-delete")
            self.summary_box = SummaryBox()
            if self.record is None or len(self.record.turns) == 0:
                self.summary_box.update(Text(self.new_label))
                self.copy_button.visible = False
            else:
                self.summary_box.update(Text(self.record.turns[-1].prompt))
            yield self.summary_box

    def update(self, record: ConversationRecord, summary: str = "") -> None:
        """Update the HistorySessionBox with a new ConversationRecord."""

        self.record = record

        # enable the copy button if it is disabled
        self.copy_button.visible = True

        if len(summary) > 0:
            self.summary_box.update(Text(summary))
            return

        if len(record.turns) == 0:
            return
        self.summary_box.update(Text(record.turns[-1].prompt))

    def on_click(self, event: Click) -> None:
        event.stop()
        self.post_message(self.Clicked(self, "load"))

    @dataclass
    class Clicked(Message):
        clicked_box: Widget
        action: str

    class ChildClicked(Clicked):
        pass

    @on(ChildClicked)
    def update_click_and_bubble(self, event: events) -> None:
        """update the widget with self and bubble up the DOM"""
        event.stop()
        self.post_message(HistorySessionBox.Clicked(self, event.action))


class DeleteButton(Static):
    """A button that deletes the session when clicked."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(":x:", *args, **kwargs)

    def on_click(self, event: Click) -> None:
        event.stop()
        self.app.log.debug("Delete button clicked")
        self.post_message(HistorySessionBox.ChildClicked(self, "delete"))


class CopyButton(Static):
    """A button that deletes the session when clicked."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(":clipboard:", *args, **kwargs)

    def on_click(self, event: Click) -> None:
        event.stop()
        self.app.log.debug("Copy button clicked")
        self.post_message(HistorySessionBox.ChildClicked(self, "copy"))


class SummaryBox(Static):
    """A summary of the current chat."""

    pass


class HistoryContainer(VerticalScroll):
    """A vertical scroll container that displays a list of Chat History sessions."""

    prompt_template = """
    Here is list of chat submissions in the form of 'user: message'. Give me no more 
    than 10 words which will be used to remind the user of the conversation.  Your reply
    should be no more than 10 words and at most 66 total characters.
    Chat submissions: {conversation}
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

    async def _add_session(
        self, record: ConversationRecord | None = None, label: str = ""
    ) -> None:
        """Add a new session to the HistoryContainer."""
        label = label if len(label) > 0 else self.new_label
        self.current = HistorySessionBox(record=record, new_label=label)
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
        conversation = "\n".join([f"user:{turn.prompt}" for turn in turns])
        # Summarize the conversation
        summary = self.llm_chain.run(conversation)

        self.current.update(record, summary)

    async def new_session(self) -> None:
        await self._add_session()

    @property
    def session_count(self) -> int:
        # query the dom for all HistorySessionBox widgets
        sessions = self.query(HistorySessionBox)
        return len(sessions)

    @on(HistorySessionBox.Clicked)
    async def history_session_box_clicked(self, event: events) -> None:
        assert event.action in ["load", "delete", "copy"]
        if event.action == "load":
            # set the clicked box to current and set the .-active class
            self.current = event.clicked_box
            self.current.add_class("-active")
            # remove the .-active class from all other boxes
            for child in self.children:
                if child != self.current:
                    child.remove_class("-active")
            self.post_message(
                HistoryContainer.HistorySessionClicked(self.current.record)
            )
            return
        if event.action == "delete":
            # if the clicked box is the only box, reset the current box
            if self.session_count == 1:
                await event.clicked_box.remove()
                await self.new_session()
                self.current.record = ConversationRecord()
                # let the app know so it can clear the chat pane
                self.post_message(
                    HistoryContainer.HistorySessionClicked(self.current.record)
                )
            elif event.clicked_box == self.current:
                # if the clicked box is the current session, set the current session
                # to the previous session or next session if there is no previous
                # session
                current = list(self.query(HistorySessionBox)).index(self.current)
                if current == 0:
                    # if the current session is the first session, set the second
                    # session to current
                    self.current = list(self.query(HistorySessionBox))[1]
                else:
                    # if the current session is not the first session, set the
                    # previous session to current
                    self.current = list(self.query(HistorySessionBox))[current - 1]

                # remove the clicked session
                await event.clicked_box.remove()
                # set the .-active class
                self.current.add_class("-active")
                # let the app know so it can update the chat pane
                self.post_message(
                    HistoryContainer.HistorySessionClicked(self.current.record)
                )
                return
            else:
                # if the clicked box is not the current session, just remove it
                await event.clicked_box.remove()
                return
        if event.action == "copy":
            new = event.clicked_box.record.copy()
            self.app.log.debug(f"orig record: {id(event.clicked_box.record)}")
            self.app.log.debug(f"new record: {id(new)}")
            await self._add_session(
                new,
                label=event.clicked_box.summary_box.renderable,
            )
            self.post_message(
                HistoryContainer.HistorySessionClicked(self.current.record)
            )

    class HistorySessionClicked(Message):
        def __init__(self, record: ConversationRecord) -> None:
            self.record = record
            super().__init__()
