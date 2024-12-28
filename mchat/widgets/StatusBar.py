from dataclasses import dataclass
from typing import List, Tuple

from textual import on
from textual.app import ComposeResult
from textual.containers import HorizontalGroup
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Label, Select


class StatusBar(Widget):
    """A status bar widget"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def compose(self) -> ComposeResult:
        with HorizontalGroup():
            yield Label("Agent:")
            # Actual values will get filled in later
            self.agent_selector = Select(
                [("", "")], allow_blank=False, id="agent_selector"
            )
            yield self.agent_selector
            yield Label("Streaming:")
            self.streaming_selector = Select(
                [("On", "On"), ("Off", "Off")],
                allow_blank=False,
                id="streaming_selector",
            )
            yield self.streaming_selector
            yield Label("Model:")
            self.model_selector = Select(
                [("", "")], allow_blank=False, id="model_selector"
            )
            yield self.model_selector

    def load_agents(
        self, agents: List[Tuple[str, str]], value: str | None = None
    ) -> None:
        """Load the agents into the agent selector widget"""
        self.agent_selector.set_options(options=agents)
        if value is not None:
            self.agent_selector.value = value

    def load_models(
        self, models: List[Tuple[str, str]], value: str | None = None
    ) -> None:
        """Load the agents into the agent selector widget"""
        self.model_selector.set_options(options=models)
        if value is not None:
            self.model_selector.value = value

    def disable_stream_selector(self) -> None:
        """Disable the streaming selector"""
        self.streaming_selector.disabled = True

    def enable_stream_selector(self) -> None:
        """Enable the streaming selector"""
        self.streaming_selector.disabled = False

    def set_streaming(self, value: bool) -> None:
        """Set the streaming selector value"""
        if value:
            self.streaming_selector.value = "On"
        else:
            self.streaming_selector.value = "Off"

    def enable_model_selector(self) -> None:
        """Enable the model selector"""
        self.model_selector.disabled = False

    def disable_model_selector(self) -> None:
        """Disable the model selector"""
        self.model_selector.disabled = True

    def set_model(self, value: str) -> None:
        """Set the model selector value"""
        self.model_selector.value = value

    @on(Select.Changed, "#agent_selector")
    def agent_changed(self, event) -> None:
        self.post_message(StatusBar.AgentChangedMessage(agent=event.value))

    @on(Select.Changed, "#model_selector")
    def model_changed(self, event) -> None:
        self.post_message(StatusBar.ModelChangedMessage(model=event.value))

    @on(Select.Changed, "#streaming_selector")
    def streaming_changed(self, event) -> None:
        if event.value == "On":
            self.post_message(StatusBar.StreamingChangedMessage(streaming=True))
        else:
            self.post_message(StatusBar.StreamingChangedMessage(streaming=False))

    @dataclass
    class AgentChangedMessage(Message):
        agent: str

    @dataclass
    class ModelChangedMessage(Message):
        model: str

    @dataclass
    class StreamingChangedMessage(Message):
        streaming: bool
