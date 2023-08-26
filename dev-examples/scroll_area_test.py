from textual.widgets import Label, Button, Markdown
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
        with VerticalScroll():
            yield Markdown("this is a start")

        yield Label("This is a label")
        self.message = ""

    def on_button_pressed(self, event: Button.Pressed) -> None:
        md_area = self.query_one(Markdown)
        self.message += "This is new text"
        md_area.update(self.message)


if __name__ == "__main__":
    app = ScrollApp()
    app.run()
