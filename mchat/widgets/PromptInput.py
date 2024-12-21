from __future__ import annotations

from dataclasses import dataclass

from textual import events, on
from textual.app import ComposeResult
from textual.css.scalar import Scalar
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Label, TextArea


class MultiLineInput(TextArea):
    """A multi-line text input widget modified from the stock TextArea widget."""

    starting_height = Scalar.from_number(3)

    DEFAULT_CSS = """
    MultiLineInput {
        height: 3;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.show_line_numbers = False
        self.scrollpad = 0
        """ Extra padding to add to the height of the TextArea to show scrollbar"""

    def _on_key(self, event) -> None:
        # if enter key was selected, submit the input
        key = event.key

        if key == "enter":
            event.prevent_default()
            self.post_message(self.Submitted(self, self.text))
            # reset height of prompt area to starting height
            self.styles.height = self.starting_height

        # if shift-down was selected, insert a newline, don't bubble the key
        elif key == "shift+down":
            self.insert("\n")
            event.prevent_default()

    @on(TextArea.Changed)
    def handle_changed(self, event: events) -> None:
        """Check if the size of the TextArea has changed and refresh if necessary."""

        # Grow to up to 8 lines before scrolling
        height = self.wrapped_document.height - 1
        if height <= 8:
            self.styles.height = Scalar.from_number(
                height + self.starting_height.cells + self.scrollpad
            )
        else:
            self.styles.height = Scalar.from_number(8 + self.starting_height.cells)

    @dataclass
    class Submitted(Message):
        """Posted when the enter key is pressed within an `Input`.

        Can be handled using `on_input_submitted` in a subclass of `Input` or in a
        parent widget in the DOM.
        """

        input: MultiLineInput
        """The `Input` widget that is being submitted."""
        value: str
        """The value of the `Input` being submitted."""


class PromptInput(Widget):
    """A multi-line text input widget modified from the stock Input widget."""

    def __init__(
        self,
        prompt: str = "Press [b]Enter[/] to submit, [b]Shift-⬇[/] for a newline, "
        + "and [b]Ctrl+C[/] to exit.",
        *args,
        **kwargs,
    ) -> None:
        self.prompt = prompt
        super().__init__(*args, **kwargs)

    def compose(self) -> ComposeResult:
        self.instructions_label = Label(
            self.prompt,
            id="instructions",
        )
        yield self.instructions_label
        self.ml_input = MultiLineInput()
        yield self.ml_input

    def focus(self) -> None:
        self.ml_input.focus()

    @property
    def value(self):
        return self.ml_input.text

    @value.setter
    def value(self, value):
        self.ml_input.load_text(value)

    @dataclass
    class Changed(Message):
        """Posted when the value changes.

        Can be handled using `on_input_changed` in a subclass of `Input` or in a parent
        widget in the DOM.
        """

        input: MultiLineInput
        """The `Input` widget that was changed."""

        value: str
        """The value that the input was changed to."""

        @property
        def control(self) -> MultiLineInput:
            """Alias for self.input."""
            return self.ml_input.text

    @dataclass
    class Submitted(Message):
        """Message to pass along the sumbit from MultiLineInput to"""

        input: MultiLineInput
        """The `Input` widget that was changed."""
        value: str
        """The value that the input was changed to."""

        @property
        def control(self) -> MultiLineInput:
            """Alias for self.input."""
            return self.ml_input.text

    @on(MultiLineInput.Submitted)
    def pass_submitted(self, event: events) -> None:
        """Pass 'submitted' events up to the DOM"""

        event.stop()
        self.post_message(PromptInput.Submitted(self, event.value))

    @on(TextArea.Changed)
    def pass_changed(self, event: events) -> None:
        """Pass changed eventus up the DOM"""
        event.stop()
        self.post_message(PromptInput.Changed(self, event.control))
