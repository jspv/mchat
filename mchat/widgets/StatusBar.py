from dataclasses import dataclass
from typing import List, Tuple

from textual import on
from textual.app import ComposeResult
from textual.containers import HorizontalGroup
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Label, Select


class StatusBar(Widget):
    """A status bar widget"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def compose(self) -> ComposeResult:
        with HorizontalGroup():
            yield Button("End", id="end_button")
            yield Button("ESC", id="escape_button")
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

    async def on_mount(self) -> None:
        self.query_one("#end_button").disabled = True
        self.query_one("#escape_button").disabled = True

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
        with self.prevent(Select.Changed):
            self.model_selector.set_options(options=models)
        if value is not None:
            self.model_selector.value = value

    @property
    def agent(self) -> str:
        """Get the current agent"""
        return self.agent_selector.value

    @agent.setter
    def agent(self, value: str) -> None:
        """Set the current agent"""
        self.agent_selector.value = value

    @property
    def model(self) -> str:
        """Get the current model"""
        return self.model_selector.value

    @model.setter
    def model(self, value: str) -> None:
        """Set the current model"""
        self.model_selector.value = value

    def disable_stream_selector(self) -> None:
        """Disable the streaming selector"""
        self.streaming_selector.value = "Off"
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

    def enable_end_button(self) -> None:
        """Enable the end button"""
        self.query_one("#end_button").disabled = False

    def disable_end_button(self) -> None:
        """Disable the end button"""
        self.query_one("#end_button").disabled = True

    def enable_escape_button(self) -> None:
        """Enable the escape button"""
        self.query_one("#escape_button").disabled = False

    def disable_escape_button(self) -> None:
        """Disable the escape button"""
        self.query_one("#escape_button").disabled = True

    @on(Button.Pressed)
    async def button_pressed(self, event) -> None:
        if event.button.id == "end_button":
            self.post_message(StatusBar.EndButtonPressedMessage())
        elif event.button.id == "escape_button":
            self.post_message(StatusBar.EscapeButtonPressedMessage())

    def enable_model_selector(self) -> None:
        """Enable the model selector"""
        self.model_selector.disabled = False

    def disable_model_selector(self) -> None:
        """Disable the model selector"""
        self.model_selector.disabled = True

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

    @dataclass
    class EndButtonPressedMessage(Message):
        pass

    @dataclass
    class EscapeButtonPressedMessage(Message):
        pass
