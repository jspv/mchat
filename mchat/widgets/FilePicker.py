from textual.widgets import DirectoryTree

from textual.app import ComposeResult
from textual.binding import Binding, _Bindings
from textual.containers import Horizontal, Container, Vertical, VerticalScroll
from textual.reactive import Reactive
from textual.widget import Widget
from textual.widgets import Button, Static, DirectoryTree
from textual.message import Message
from textual.events import Click
from textual import on
from dataclasses import dataclass


from pathlib import Path
from typing import Iterable


class FilteredDirectoryTree(DirectoryTree):
    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        """Filter paths to only show certain files"""
        return [
            path
            for path in paths
            if path.name.endswith(".pdf") or path.name.endswith(".txt") or path.is_dir()
        ]

    def __init__(self, path: Path, **kwargs):
        absolute_path = str(Path(path).absolute())
        super().__init__(path=absolute_path, **kwargs)


class DirUpButton(Static):
    """A button that reorients the Tree up one level."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(":upwards_button:", *args, **kwargs)

    def on_click(self, event: Click) -> None:
        event.stop()
        self.app.log.debug("DirUp button clicked")
        self.post_message(FilePickerDialog.ChildClicked(self, "dirup"))


class FilePickerDialog(Widget):
    """Display a modal dialog"""

    _show_dialog = Reactive(False)
    # DEFAULT_CSS = """

    # /* The top level dialog (a Container) */

    # """

    BINDINGS = [
        ("y", "run_confirm_binding('dialog_y')", "OK"),
        ("n", "run_confirm_binding('dialog_n')", "Cancel"),
        ("u", "go_up_tree", "DirUp"),
        ("right", "focus_on_directory", "DirFocus"),
    ]

    def __init__(
        self,
        confirm_action: str | None = None,
        noconfirm_action: str | None = None,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ):
        self.confirm_action = confirm_action
        self.noconfirm_action = noconfirm_action
        # actual message will be set using set_message()
        self.message = ""

        # list to save and restore focus for modal dialogs
        self._focuslist = []
        self._focus_save = None
        self._bindings_stack = []

        super().__init__(name=name, id=id, classes=classes)

    class FocusMessage(Message):
        """Message to inform the app that Focus has been taken"""

        def __init__(self, focustaken=True) -> None:
            self.focustaken = focustaken
            super().__init__()

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static(self.message, id="file_picker_dialog_question"),
            VerticalScroll(DirUpButton(), FilteredDirectoryTree(".")),
            Horizontal(
                Button("OK", variant="success", disabled=True, id="dialog_y"),
                Button("Cancel", variant="error", id="dialog_n"),
                classes="file_picker_dialog_buttons",
            ),
            id="file_picker_dialog",
        )

    @dataclass
    class ChildClicked(Message):
        """Message from a child inform the app that a child widget has been clicked"""

        clicked_box: Widget
        action: str

    @on(ChildClicked)
    async def _handle_child_clicked(self, message: ChildClicked):
        """Handle a child widget being clicked"""
        if message.action == "dirup":
            await self.run_action("go_up_tree")

    def on_directory_tree_directory_selected(
        self, event: DirectoryTree.DirectorySelected
    ) -> None:
        """When a directory is selected, enable the DirFocus binding"""

        self.app.log.debug(f"Directory selected: {event.path}")
        self.selected_path = event.path
        button = self.query_one("#dialog_y", Button)
        button.disabled = True

    def on_directory_tree_file_selected(
        self, event: DirectoryTree.FileSelected
    ) -> None:
        """When a file is selected, enable the confirm button"""
        self.app.log.debug(f"File selected: {event.path}")
        self.selected_path = event.path
        button = self.query_one("#dialog_y", Button)
        button.disabled = False

    def watch__show_dialog(self, show_dialog: bool) -> None:
        """Called when _show_dialog is modified"""
        self.app.set_class(show_dialog, "-show-file_picker_dialog")

    def set_message(self, message: str) -> None:
        """Update the dialgo message"""
        self.query_one("#file_picker_dialog_question", Static).update(message)

    def show_dialog(self) -> None:
        self._override_bindings()
        self._override_focus()
        self._show_dialog = True

    def action_close_dialog(self) -> None:
        """Close the dialog and return bindings"""
        self._restore_bindings()
        self._restore_focus()
        self._show_dialog = False

    def action_go_up_tree(self) -> None:
        """Go up a directory"""
        tree = self.query_one(FilteredDirectoryTree)
        tree.path = tree.path.parent
        tree.reload()

    def action_focus_on_directory(self) -> None:
        """Focus on the directory tree"""
        # if the selected path is a directory, focus on it
        if self.selected_path.is_dir():
            tree = self.query_one(FilteredDirectoryTree)
            tree.path = self.selected_path
            tree.reload()

    @property
    def confirm_action(self):
        return self._confirm_action

    @confirm_action.setter
    def confirm_action(self, value: str):
        self._confirm_action = value

    @property
    def noconfirm_action(self):
        return self._noconfirm_action

    @noconfirm_action.setter
    def noconfirm_action(self, value: str):
        self._noconfirm_action = value

    def _override_bindings(self):
        """Force bindings for the dialog"""
        self._bindings_stack.append(self.app._bindings)
        newbindings = [
            Binding(
                key="ctrl+c",
                action="quit",
                description="",
                show=False,
                key_display=None,
                priority=True,
            ),
        ]
        self.app._bindings = _Bindings(newbindings)

    async def _action_run_confirm_binding(self, answer: str):
        """When someone presses a button or key, directly run the associated binding"""
        if answer == "dialog_y":
            # if the button is disabled, do nothing
            if self.query_one("#dialog_y", Button).disabled:
                return
            await self.run_action(self._confirm_action)
            if self._confirm_action != "close_dialog":
                self.action_close_dialog()
        elif answer == "dialog_n":
            # if the button is disabled, do nothing
            if self.query_one("#dialog_n", Button).disabled:
                return
            await self.run_action(self._noconfirm_action)
            if self._confirm_action != "close_dialog":
                self.action_close_dialog()
        else:
            raise ValueError

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        assert button_id is not None
        await self._action_run_confirm_binding(button_id)

    def _restore_bindings(self):
        if len(self._bindings_stack) > 0:
            self.app._bindings = self._bindings_stack.pop()

    def _override_focus(self):
        """remove focus for everything, force it to the dialog"""
        self._focus_save = self.app.focused
        for widget in self.app.screen.focus_chain:
            self._focuslist.append(widget)
            widget.can_focus = False
        self.can_focus = True
        self.focus()
        self.post_message(self.FocusMessage(focustaken=True))

    def _restore_focus(self):
        """restore focus to what it was before we stole it"""
        while len(self._focuslist) > 0:
            self._focuslist.pop().can_focus = True
        if self._focus_save is not None:
            self.app.set_focus(self._focus_save)
        self.post_message(self.FocusMessage(focustaken=False))
