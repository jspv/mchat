import asyncio
from typing import Callable

from rich.console import RenderableType
from textual.containers import VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Collapsible, Static


class DebugPane(VerticalScroll):
    _entries = {}

    async def add_entry(
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
        if asyncio.iscoroutinefunction(value):
            value = str(await value())

        self.mount(
            Collapsible(
                Static(str(value), id="debug-" + key), title=keymsg, collapsed=collapsed
            )
        )

    async def update_entry(
        self,
        key: str,
        value: RenderableType | Callable = None,
    ) -> None:
        (keymsg, old_value) = self._entries[key]

        if value is None:
            value = old_value

        self._entries[key] = (keymsg, value)
        if asyncio.iscoroutinefunction(value):
            value = str(await value())
        if isinstance(value, Callable):
            value = str(value())
        try:
            widget = self.query_one("#debug-" + key)
        except NoMatches:
            # this will happen when the application is exiting
            pass
        else:
            widget.update(value)

    async def update_status(self) -> None:
        """force debug pane to update all values"""

        for key, (keymsg, value) in self._entries.items():  # noqa: B007
            await self.update_entry(key, value)
