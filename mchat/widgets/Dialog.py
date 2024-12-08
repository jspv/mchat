from textual.screen import ModalScreen
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, Static


class Dialog(ModalScreen):
    """Display a modal dialog"""

    DEFAULT_CSS = """

        Dialog Container {
            border: round $primary;
            box-sizing: border-box;
            background: $boost;
            width: auto;
            height: auto;
        }

        /* Matches the question text in the button */
        #dialog_question {
            text-style: bold;
            width: 100%;
            height: auto;
            content-align: center middle;
            margin: 0 0 2 0;
        }

        /* Matches the container holding the dialog buttons */
        .dialog_buttons {
            align: center middle;
            height: auto;
            width: auto;
            /* border: wide red; */
        }

        /* The button class */
        Dialog Button {
            margin: 0 4;
        }

    """

    BINDINGS = [
        ("y", "run_confirm_binding('dialog_y')", "Yes"),
        ("n", "run_confirm_binding('dialog_n')", "No"),
    ]

    def __init__(
        self,
        confirm_action: str | None = None,
        noconfirm_action: str | None = None,
        name: str | None = None,
        id: str | None = "modal_dialog",
        classes: str | None = None,
    ):
        self.confirm_action = confirm_action
        self.noconfirm_action = noconfirm_action
        # actual message will be set using set_message()
        self._message = "This is a modal dialog"
        super().__init__(name=name, id=id, classes=classes)

    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Static(self._message, id="dialog_question")
            with Horizontal(classes="dialog_buttons"):
                yield Button("Yes", variant="success", id="dialog_y")
                yield Button("No", variant="error", id="dialog_n")

    def set_message(self, message: str) -> None:
        """Update the dialog message"""
        self._message = message
        # self.query_one("#dialog_question", Static).update(message)

    def show_dialog(self) -> None:
        pass

    def action_dialog_close(self) -> None:
        """Close the dialog and return bindings"""
        self.app.pop_screen()

    @property
    def confirm_action(self):
        return self._confirm_action

    @confirm_action.setter
    def confirm_action(self, value: str):
        # if there is no namespace set and action doesn't start with 'dialog_'
        # assume app namespace
        if value is not None and not value.startswith("dialog_"):
            value = f"app.{value}"

        self._confirm_action = value

    @property
    def noconfirm_action(self):
        return self._noconfirm_action

    @noconfirm_action.setter
    def noconfirm_action(self, value: str):
        # if there is no namespace set, assume app
        if value is not None and not value.startswith("dialog_"):
            value = f"app.{value}"
        self._noconfirm_action = value

    async def _action_run_confirm_binding(self, answer: str):
        """When someone presses a button, directly run the associated binding"""
        if answer == "dialog_y":
            await self.run_action(self._confirm_action)
        elif answer == "dialog_n":
            await self.run_action(self._noconfirm_action)
        else:
            raise ValueError

    async def _on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        assert button_id is not None
        await self._action_run_confirm_binding(button_id)
