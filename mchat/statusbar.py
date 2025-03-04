from nicegui import ui


class StatusContainer:
    def __init__(self, app):
        self.app = app
        with ui.row().classes("items-center"):
            self.text = ui.label("Agent:").classes("text-[#121212] dark:text-[#ffffff]")
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
                .classes("text-[#121212] dark:text-[#ffffff]")
                .props("color=secondary left-label dense")
                .bind_value(self.app.ag, "stream_tokens")
                .bind_enabled_from(self.app, "ui_is_busy", backward=lambda x: not x)
            )
            self.app.logger.debug(f"streaming: {self.app.ag.stream_tokens}")
            self.text = ui.label("Model:").classes("text-[#121212] dark:text-[#ffffff]")
            self.models()

    @ui.refreshable
    def models(self):
        self.model = (
            ui.select(self.app.current_compatible_models)
            .bind_value(self.app, "current_llm_model")
            .bind_enabled_from(self.app, "ui_is_busy", backward=lambda x: not x)
            .props("color=secondary rounded standout dense")
        )
