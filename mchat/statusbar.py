import logging

from nicegui import ui

from .styles import colors as c

logger = logging.getLogger(__name__)


class StatusContainer:
    def __init__(self, app):
        self.app = app
        with ui.row().classes("items-center"):
            self.text = ui.label("Agent:").classes(
                f"text-{c.darkpage} dark:text-{c.lightpage}"
            )
            self.agent = (
                ui.select(self.app.chooseable_agents, value=self.app.current_agent)
                .bind_value(self.app, "current_agent")
                .bind_enabled_from(self.app, "ui_is_busy", backward=lambda x: not x)
                .on_value_change(lambda: self.models.refresh())
                .props("color=secondary rounded standout dense")
            )
            # ui.query(f"#c{self.agent.id}").classes("text-red-500")
            self.streaming = (
                ui.switch("Streaming", value=False)
                .classes(f"text-{c.darkpage} dark:text-{c.lightpage}")
                .props("color=secondary left-label dense")
                .bind_value(self.app.ag, "stream_tokens")
                .bind_enabled_from(self.app, "ui_is_busy", backward=lambda x: not x)
            )
            logger.debug(f"streaming: {self.app.ag.stream_tokens}")
            self.text = ui.label("Model:").classes(
                f"text-{c.darkpage} dark:text-{c.lightpage}"
            )
            self.models()

    @ui.refreshable
    def models(self):
        self.model = (
            ui.select(self.app.current_compatible_models)
            .bind_value(self.app, "current_llm_model")
            .bind_enabled_from(self.app, "ui_is_busy", backward=lambda x: not x)
            .props("color=secondary rounded standout dense")
        )
