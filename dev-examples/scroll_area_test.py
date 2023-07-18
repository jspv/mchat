from textual.widgets import Label, Button
from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer, VerticalScroll
from textual.reactive import reactive


class ChatBox(Label):
    _label_text: reactive[str] = reactive("", repaint=False)
    """This is used as an auxiliary reactive to only refresh the label when needed."""

    def watch__label_text(self, label_text: str) -> None:
        """If the label text changed, update the renderable (which also refreshes)."""
        self.update(label_text)


class ScrollApp(App):
    def compose(self) -> ComposeResult:
        yield Button("Start", variant="success")
        yield VerticalScroll(
            Label("This is a label 01"),
            Label("This is a label 02"),
            Label("This is a label 03"),
            Label("This is a label 04"),
            Label("This is a label 05"),
            Label("This is a label 06"),
            Label("This is a label 07"),
            Label("This is a label 08"),
            Label("This is a label 09"),
            Label("This is a label 10"),
            Label("This is a label 11"),
            Label("This is a label 12"),
            Label("This is a label 13"),
            Label("This is a label 14", id="foo"),
            Label("This is a label 15"),
            Label("This is a label 16"),
            Label("This is a label 17"),
            Label("This is a label 18"),
            Label("This is a label 19"),
            Label("This is a label 20"),
            Label("This is a label 21"),
            Label("This is a label 22"),
        )
        yield Label("This is a label")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        label = self.query_one("#foo")
        label_text = label.renderable.markup
        label_text += " Add On"
        label.update(label_text)


if __name__ == "__main__":
    app = ScrollApp()
    app.run()
