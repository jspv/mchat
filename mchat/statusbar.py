from nicegui import ui


class StatusContainer:
    def __init__(self, app):
        self.app = app
        with ui.row().classes("items-center"):
            self.text = ui.label("Agent:")
            self.agent = (
                ui.select(self.app.chooseable_agents, value=self.app.current_agent)
                .bind_value(self.app, "current_agent")
                .on_value_change(lambda: self.models.refresh())
                .props("color=secondary rounded standout")
            )
            ui.query(f"#c{self.agent.id}").classes("text-red-500")
            self.streaming = (
                ui.switch("Streaming", value=False)
                .props("color=secondary left-label")
                .bind_value(self.app.ag, "stream_tokens")
            )
            self.app.logger.debug(f"streaming: {self.app.ag.stream_tokens}")
            self.text = ui.label("Model:")
            self.models()

    @ui.refreshable
    def models(self):
        self.model = (
            ui.select(self.app.current_compatible_models)
            .bind_value(self.app, "current_llm_model")
            .props("color=secondary rounded standout")
        )
