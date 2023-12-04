from textual.widgets import Static, Collapsible
from textual.reactive import Reactive
from textual.containers import VerticalScroll
from textual.css.query import NoMatches
from rich.console import RenderableType
from rich.text import Text
from typing import Callable
from rich.padding import Padding


class DebugPane(VerticalScroll):
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
        try:
            widget = self.query_one("#debug-" + key)
        except NoMatches:
            # this will happen when the application is exiting
            pass
        else:
            widget.update(value)

    def update_status(self) -> None:
        """force debug pane to update all values"""

        for key, (keymsg, value) in self._entries.items():
            self.update_entry(key, value)
