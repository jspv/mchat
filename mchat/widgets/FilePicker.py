from typing import List
from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, Static, DirectoryTree
from textual.message import Message
from textual.events import Click
from textual.css.query import NoMatches
from textual import on
from dataclasses import dataclass


from pathlib import Path
from typing import Iterable


class FilteredDirectoryTree(DirectoryTree):
    """A DirectoryTree that can filter files"""

    extensions = []
    show_dirs = True

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        """Filter paths to only show certain files"""

        returned = []
        for path in paths:
            if self.show_dirs is True and path.is_dir():
                returned.append(path)
            elif not self.extensions:
                returned.append(path)
            elif path.suffix in self.extensions:
                returned.append(path)
        return returned

    def __init__(self, path: Path, extensions: List[str] = [], **kwargs):
        absolute_path = str(Path(path).absolute())
        self.extensions = extensions
        super().__init__(path=absolute_path, **kwargs)


class DirUpButton(Static):
    """A button that reorients the Tree up one level."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(":upwards_button:", *args, **kwargs)

    def on_click(self, event: Click) -> None:
        event.stop()
        self.app.log.debug("DirUp button clicked")
        self.post_message(FilePickerDialog.ChildClicked(self, "dirup"))


class FilePickerDialog(ModalScreen):
    """Display a modal dialog"""

    DEFAULT_CSS = """

        FilePickerDialog Container{
            border: round $primary;
            box-sizing: border-box;
            background: $boost;
            height: auto;
            max-height: 80%;
            width: auto;
        }

        FilePickerDialog Horizontal {
            width: auto;
            height: auto;
        }

        FilePickerDialog DirUpButton {
            width: auto;
            height: auto;
        }

        FilePickerDialog Vertical {
            width: auto;
            height: auto;
        }

        /* Matches the question text  */
        #file_picker_dialog_question {
            text-style: bold;
            width: auto;
            height: auto;
            content-align: center middle;
            margin: 0 0 0 0;
        }

        FilePickerDialog VerticalScroll {
            border: white;
            width: auto;
            height: 1fr;
        }

        DirectoryTree {
            width: auto;
            height: auto;
            min-width: 45;
        }

        /* Matches the Horizontal container holding the dialog buttons */
        .file_picker_dialog_buttons {
            align: center middle;
            width: 100%;
        }

        /* The button class */
        FilePickerDialog Button {
            margin: 0 4;
        }

    """

    BINDINGS = [
        ("y", "run_confirm_binding('dialog_y')", "OK"),
        ("n", "run_confirm_binding('dialog_n')", "Cancel"),
        ("u", "go_up_tree", "DirUp"),
        ("right", "focus_on_directory", "DirFocus"),
    ]

    current_path = Path.cwd()

    def __init__(
        self,
        confirm_action: str | None = None,
        noconfirm_action: str | None = None,
        name: str | None = None,
        id: str | None = "file_picker",
        extensions: List[str] = [],
        classes: str | None = None,
    ):
        self.confirm_action = confirm_action
        self.noconfirm_action = noconfirm_action
        self.extensions = extensions
        # actual message will be set using set_message()
        self._message = ""

        super().__init__(name=name, id=id, classes=classes)

    def compose(self) -> ComposeResult:
        with Container(id="file_picker_dialog"):
            with Vertical():
                yield Static(self.message, id="file_picker_dialog_question")
                with VerticalScroll():
                    yield DirUpButton()
                    yield FilteredDirectoryTree(
                        self.current_path, extensions=self.extensions
                    )
                with Horizontal(classes="file_picker_dialog_buttons"):
                    yield Button("OK", variant="success", disabled=True, id="dialog_y")
                    yield Button("Cancel", variant="error", id="dialog_n")

    # Custom Messages

    @dataclass
    class ChildClicked(Message):
        """Message to inform this dialog that a child widget has been clicked"""

        clicked_box: Widget
        action: str

    # message handlers

    @on(ChildClicked)
    async def _handle_child_clicked(self, message: ChildClicked):
        """Handle a child widget being clicked and run the associated action"""
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

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """When a button is pressed, run the associated action"""
        button_id = event.button.id
        assert button_id is not None
        await self._action_run_confirm_binding(button_id)

    def watch__show_dialog(self, show_dialog: bool) -> None:
        """Called when _show_dialog is modified"""
        self.app.set_class(show_dialog, "-show-file_picker_dialog")

    @property
    def message(self):
        return self._message

    @message.setter
    def message(self, value: str):
        """Update the dialog message"""
        self._message = value
        # self.query_one("#file_picker_dialog_question", Static).update(self._message)

    def show_dialog(self) -> None:
        """Show the filepicker"""
        # reload the tree as filter may have changed
        tree = self.query_one(FilteredDirectoryTree)
        tree.path = self.current_path
        tree.reload()

    def action_close_dialog(self) -> None:
        """Close the dialog and return bindings"""
        self.app.pop_screen()

    def _action_go_up_tree(self) -> None:
        """Go up a directory, but only if directory traversal is enabled"""
        if self.show_dirs:
            tree = self.query_one(FilteredDirectoryTree)
            tree.path = tree.path.parent
            tree.reload()

    def _action_focus_on_directory(self) -> None:
        """Focus on the directory tree"""
        # if the selected path is a directory, focus on it
        if self.selected_path.is_dir():
            tree = self.query_one(FilteredDirectoryTree)
            tree.path = self.selected_path
            self.current_path = self.selected_path
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

    @property
    def path(self):
        return self.current_path

    @path.setter
    def path(self, value: Path):
        if value.is_dir():
            self.current_path = value
        else:
            self.current_path = value.parent
        # reload the tree
        tree = self.query_one(FilteredDirectoryTree)
        tree.path = self.current_path
        tree.reload()

    @property
    def allowed_extensions(self):
        tree = self.query_one(FilteredDirectoryTree)
        return tree.extensions

    @allowed_extensions.setter
    def allowed_extensions(self, value: list[str]):
        self.extensions = value
        try:
            tree = self.query_one(FilteredDirectoryTree)
        # the tree may not be mounted yet, if not, do nothing it will get the
        # extensions via self.extesions whne it is mounted
        except NoMatches:
            return
        tree.extensions = value
        tree.reload()
        # set both the dialog and tree extensions

    @property
    def show_dirs(self):
        tree = self.query_one(FilteredDirectoryTree)
        return tree.show_dirs

    @show_dirs.setter
    def show_dirs(self, value: bool):
        tree = self.query_one(FilteredDirectoryTree)
        tree.show_dirs = value
        tree.reload()

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
