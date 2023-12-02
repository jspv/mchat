from textual.widget import Widget
from textual.widgets import Static, Collapsible, Label
from textual.reactive import Reactive
from textual.containers import VerticalScroll
from rich.console import RenderableType
from rich.text import Text
from typing import Callable
from rich.padding import Padding


class DebugPane(VerticalScroll):
    message: RenderableType | None = None
    _regenerate_from_dict: Reactive(bool) = Reactive(False)
    _entries = {}

    def add_entry(
        self,
        key: str,
        keymsg: RenderableType,
        value: RenderableType | Callable,
        collapsed: bool = False,
    ) -> None:
        self._entries[key] = (keymsg, value)
        # if value is callable, call it to get the value
        if isinstance(value, Callable):
            value = str(value())
        self.mount(
            Collapsible(
                Static(str(value), id="debug-" + key), title=keymsg, collapsed=collapsed
            )
        )

    def update_entry(
        self,
        key: str,
        value: RenderableType | Callable = None,
    ) -> None:
        (keymsg, old_value) = self._entries[key]

        if value is None:
            value = old_value

        self._entries[key] = (keymsg, value)
        if isinstance(value, Callable):
            value = str(value())
        widget = self.query_one("#debug-" + key)
        widget.update(value)

    def update_status(self) -> None:
        """force debug pane to update all values"""

        for key, (keymsg, value) in self._entries.items():
            self.update_entry(key, value)

    # def compose(self) -> None:
    #     with VerticalScroll():
    #         self.debug_pane_text = DebugPaneText(id="debug-pane-text")
    #         yield self.debug_pane_text

    # def render(self) -> None:
    #     if self._regenerate_from_dict is True:
    #         self.debug_pane_text._entries = self._entries
    #         self.debug_pane_text.update()
    #         self._regenerate_from_dict = False
    #     return super().render()


class DebugPaneText(Static):
    """Text area in the Debug Pane"""

    _entries = {}

    # Component classes typically begin with the name of the widget followed
    # by two hyphens. This is a convention to avoid potential name clashes.
    # They are used to allow styling to be applied to specific components of # a widget.
    COMPONENT_CLASSES = {"debugpanetext--key", "debugpanetext--value"}

    def render(self) -> RenderableType:
        key_style = self.get_component_rich_style("debugpanetext--key")
        value_style = self.get_component_rich_style("debugpanetext--value")
        # build the text container
        self.message = Text(no_wrap=False, overflow="ellipsis", justify="left", end="")
        # Build the statuses
        for keymsg, value in self._entries.values():
            self.message.append_text(Text(keymsg, style=key_style))
            self.message.append(": ")
            if isinstance(value, Callable):
                value = value()
            value = Text(str(value), style=value_style)
            self.message.append_text(value)
            self.message.append("\n")
        return Padding(self.message, pad=(0, 1, 0, 2), expand=True)
