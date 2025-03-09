import argparse
import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Literal

import yaml
from nicegui import app, events, ui
from pygments.formatters import HtmlFormatter

from config import settings
from mchat.history import HistoryContainer
from mchat.llm import AutogenManager, ModelManager
from mchat.statusbar import StatusContainer
from mchat.styles import colors as c

plogger = logging.getLogger(__name__).parent
plogger.setLevel("WARNING")
log_handler = logging.FileHandler("debug.log", mode="a")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log_handler.setFormatter(formatter)
plogger.addHandler(log_handler)

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


logging.getLogger("mchat.llm").setLevel("DEBUG")

DEFAULT_AGENT_FILE = "mchat/default_agents.yaml"
EXTRA_AGENTS_FILE = settings.get("extra_agents_file", None)


class LogElementHandler(logging.Handler):
    """A logging handler that emits messages to a log element."""

    LOG_LEVEL_ICONS = {
        logging.DEBUG: "🐞",
        logging.INFO: "ℹ️",
        logging.WARNING: "⚠️",
        logging.ERROR: "❌",
        logging.CRITICAL: "🔥",
    }

    def __init__(self, element: ui.log, level: int = logging.NOTSET) -> None:
        self.element = element
        super().__init__(level)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            msg = f"{self.LOG_LEVEL_ICONS.get(record.levelno, '')} {msg}"
            self.element.push(msg)
        except Exception:
            self.handleError(record)


class ChatTurn(object):
    """A single turn in a chat conversation, UI and logic"""

    def __init__(
        self,
        container: ui.element,
        question: str = None,
        role: str = "user",
        title: str = "Agent",
    ):
        self.role = role
        self.agent = title  # The agent name
        self.response: str = ""
        self.container = container

        with self.container:
            if question and role == "user":
                with ui.row().classes("mt-4 mb-1 justify-end"):
                    ui.label(f"{question}").classes(
                        f"bg-{c.input_d} text-white p-4 dark rounded-3xl text-body1"
                    )
            with ui.element("div") as self.chat_response:
                self.chat_response_label = ui.label("").classes("text-[8px]")
                self.chat_response_content = ui.element("div").classes(
                    "bg-transparent p-2 text-body1"
                )
            # this keeps the area invisible until we get content
            self.chat_response.visible = False

        # force scroll to the bottom
        ui.run_javascript("window.scrollTo(0, document.body.scrollHeight)")

    async def append_chunk(self, chunk: str):
        self.response += chunk
        self.chat_response_content.clear()
        if self.chat_response.visible is False:
            self.chat_response_label.text = f"{self.agent}"
            self.chat_response.visible = True
        with self.chat_response_content:
            ui.markdown(
                self.response,
                extras=["fenced-code-blocks", "tables", "code-friendly", "latex"],
            )

        # force scroll to the bottom
        ui.run_javascript("window.scrollTo(0, document.body.scrollHeight)")


class WebChatApp:
    def __init__(self) -> None:
        # will be set by set_agent_and_model
        self._current_agent: str | None = None
        self._current_compatible_models: list[str] | None = None
        self._current_llm_model: str | None = None

        self.current_temperature: float | None = None
        self.current_question: str | None = None

        self.history_container: HistoryContainer | None = None
        self.message_container: ui.element | None = None
        self.status_container: StatusContainer | None = None
        self.chatbox: ChatTurn | None = None

        self.ui_is_busy = False

        # parse arguments
        self.parse_args_and_initialize()
        # load agents
        self.agents = self.load_agents()
        self.chooseable_agents = [
            agent_name
            for agent_name, val in self.agents.items()
            if val.get("choosable", True)
        ]

        # Get an object to manage the AI models
        self.mm = ModelManager()

        # Get an object to manage autogen
        self.ag = AutogenManager(
            message_callback=self.on_llm_new_token,
            agents=self.agents,
        )
        # load the llm models from settings - name: {key: value,...}

        self.available_llm_models = self.mm.available_chat_models
        self.available_image_models = self.mm.available_image_models

        self._current_llm_model = self.mm.default_chat_model
        self.llm_temperature = self.mm.default_chat_temperature

        # Initialize the image model
        self.default_image_model = settings.get("defaults.image_model", None)
        if self.mm.default_image_model is not None:
            self.image_model_name = self.mm.default_image_model
            self.image_model = self.mm.open_model(
                self.image_model_name, model_type="image"
            )
        else:
            self.image_model_name = None
            self.image_model = None

        # Initialze the agent, model and active conversation record

        # set the agent and model
        self.default_agent = getattr(settings, "defaults.agent", "default")

        # Define the main UI layout
        @ui.page("/")
        async def main_ui():
            await self.set_agent_and_model(
                agent=self.default_agent,
                model=self._current_llm_model,
                temperature=self.llm_temperature,
                model_context=None,
            )

            # the queries below are used to expand the contend down to the footer
            # (content can then use flex-grow to expand)
            ui.query(".q-page").classes("flex")
            ui.query(".nicegui-content").classes("w-full")
            dark_mode = ui.dark_mode(True)
            ui.colors(
                secondary="#26A69A",
                # accent="#dd4b39",
                accent="#2e2e2e",
                dark="#171717",
                positive="#21BA45",
                negative="#C10015",
                info="#31CCEC",
                warning="#F2C037",
                # primary="#212121",
                darkpage="#121212",
                lightpage="#FFFFFF",
                historycard="#2e2e2e",
            )

            # Below allows code blocks in markdown to look nice
            ui.add_head_html(
                f'<style>{HtmlFormatter(nobackground=False, style="solarized-dark").get_style_defs(".codehilite")}</style>'  # noqa: E501
            )

            ui.add_head_html("""
            <style>
                .q-btn:disabled {
                    background-color: grey !important;
                    opacity: 0.3 !important;
                }
            </style>
            """)

            self.message_container = ui.element("div").classes(
                "w-full max-w-4xl mx-auto flex-grow items-stretch"
            )

            # Header
            with ui.header(elevated=True).classes(
                f"flex items-center justify-between bg-{c.lightpage} "
                f"dark:bg-{c.darkpage}  p-1"
            ):
                ui.label("MChat").classes(
                    f"text-{c.darkpage} dark:text-{c.lightpage} text-lg"
                )
                self.status_container = StatusContainer(app=self)

                with ui.row().classes("gap-2"):
                    ui.button(icon="dark_mode", on_click=dark_mode.enable).props(
                        "outline round"
                    ).tooltip("Dark Mode").bind_visibility_from(
                        dark_mode, "value", value=False
                    )
                    ui.button(icon="light_mode", on_click=dark_mode.disable).props(
                        "outline round"
                    ).tooltip("Light Mode").bind_visibility_from(
                        dark_mode, "value", value=True
                    )

                    ui.button(on_click=lambda: right_drawer.toggle(), icon="adb").props(
                        "flat"
                    )

            # Left drawer History Container, this will aslo load the database and
            # populate the history container

            self.history_container = HistoryContainer(
                new_record_callback=self.load_record, new_label="New Session", app=self
            )
            # Load the active conversation record
            self.record = self.history_container.active_record

            # Right Drawer Debug Pane
            with (
                ui.right_drawer(value=False, bottom_corner=True, fixed=True).props(
                    "bordered"
                ) as right_drawer
                # .style("background-color: #ebf1fa")
            ):
                ui.label("DEBUG")
                # self._debug_container = ui.scroll_area().classes("h-full p-4")
                self.log = ui.log().classes("w-full whitespace-pre-wrap h-full")
                handler = LogElementHandler(self.log)
                logging.getLogger(__name__).parent.addHandler(handler)
                ui.context.client.on_disconnect(
                    lambda: logging.getLogger(__name__).parent.removeHandler(handler)
                )
                logger.info("Initialized UI Logging")

            # Footer and Input Area
            with ui.footer().classes(
                f"bg-{c.lightpage} dark:bg-{c.darkpage} p-2 justify-center pt-0"
            ):
                with ui.card().classes(
                    "bg-accent rounded-3xl p-3 px-4 w-4/5 min-h-20 flex flex-col"
                ) as card:
                    with ui.column().classes(
                        f"bg-{c.input_d} dark:bg-{c.input_d} gap-0 w-full"
                    ):
                        with (
                            ui.textarea(placeholder="How can I help?")
                            .props(
                                f"dark autogrow borderless standout='bg-{c.input_d}' "
                                f"input-class='max-h-40 bg-{c.input_d} dark:bg-{c.input_d} text-white' "
                                "dense autofocus"
                            )
                            .classes("w-full self-center text-body1")
                            .bind_enabled_from(
                                self, "ui_is_busy", backward=lambda x: not x
                            ) as input_area
                        ):

                            async def _enter(e: events.GenericEventArguments):
                                if e.args["shiftKey"]:
                                    return
                                else:
                                    # this is a new question
                                    question = input_area.value
                                    input_area.set_value("")
                                    await asyncio.sleep(0.1)  # Fix to give ui time
                                    self.ui_is_busy = True

                                    with self.message_container:
                                        self._spinner = ui.spinner(color="secondary")
                                        await self.add_to_chat_message(
                                            role="user", message=question
                                        )
                                    await self.ask_question(question)
                                    # spinner may be gone if the session changed
                                    if self._spinner.is_deleted is False:
                                        self._spinner.delete()
                                    self.ui_is_busy = False

                            input_area.on(
                                "keypress.enter",
                                _enter,
                                args=["shiftKey"],
                            )

                        # button row
                        with ui.row(align_items="end").classes(
                            "w-full items-end justify-end items-stretch"
                        ):
                            self.end_button = (
                                ui.button(icon="stop", color="warning")
                                .classes("p-0")
                                .props("outline round dense")
                                .on_click(self.end_button_pressed)
                                .bind_visibility_from(self, "ui_is_busy")
                                .tooltip("Stop the agent on next turn")
                            )

                            self.esc_button = (
                                ui.button(icon="cancel", color="red")
                                .props("outline round dense")
                                .classes("p-0")
                                .on_click(self.esc_button_pressed)
                                .bind_visibility_from(self, "ui_is_busy")
                                .tooltip("Immediately terminate the agent")
                            )

                            self.submit_button = (
                                ui.button(icon="arrow_upward")
                                .props("outline round dense")
                                .on("click", _enter, args=["shiftKey"])
                                .bind_visibility_from(
                                    self, "ui_is_busy", backward=lambda x: not x
                                )
                            )

                        # focus the input when anywhere on the card is clicked
                        card.on("click", lambda: input_area.run_method("focus"))

        logger.info("Starting MChat")
        logger.debug("Debug logging enabled")

    @property
    def current_agent(self):
        return self._current_agent

    @current_agent.setter
    def current_agent(self, agent: str):
        logger.info(f"setting agent to {agent}")
        asyncio.create_task(self.set_agent_and_model(agent=agent))

    @property
    def current_compatible_models(self):
        return self._current_compatible_models

    @current_compatible_models.setter
    def current_compatible_models(self, models: list[str]):
        self._current_compatible_models = models
        if self.status_container is not None:
            self.status_container.models.refresh()

    @property
    def current_llm_model(self):
        return self._current_llm_model

    @current_llm_model.setter
    def current_llm_model(self, model: str):
        if model is None:  # this can happen when the UI is refreshing
            return
        if model not in self.current_compatible_models:
            raise ValueError(
                f"model '{model}' not compatible with agent '{self.agent}'"
            )
        logger.info(f"setting model to {model}")
        asyncio.create_task(self.set_agent_and_model(model=model))

    def handle_exception(self, e: Exception) -> None:
        """callbacks don't propogate excecptions, so this sends them to ui.notify"""
        logger.warning(f"Exception: {e}")
        ui.notify(f"Exception: {e}", type="warning")

    def run(self, **kwargs):
        # callbacks don't propagate exceptions, so this sends them to ui.notify
        app.on_exception(self.handle_exception)
        logger.info(f"Starting MChat at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        ui.run(**kwargs)

    async def on_llm_new_token(self, token: str, **kwargs):
        """Callback for new tokens from the autogen manager"""
        # if an agent is set, note it in the message
        agent_name = kwargs.get("agent", None)

        # if the token is empty, don't post it
        if not token:
            return

        await self.add_to_chat_message(
            role="assistant", message=token, agent_name=agent_name
        )
        # if this is marked as a complete message, end the turn
        if "complete" in kwargs and kwargs["complete"]:
            await self.end_chat_turn(role="assistant", agent_name=agent_name)

    def set_agent(self, agent: str):
        """Callback to set the agent"""
        logger.info(f"Setting agent to {agent}")
        asyncio.create_task(self.set_agent_and_model(agent=agent))

    def parse_args_and_initialize(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-v", "--verbose", help="Increase verbosity", action="store_true"
        )
        args, unknown = parser.parse_known_args()

    async def add_to_chat_message(
        self, message: str, role: str | None = None, agent_name: str = None
    ):
        # if the agent_name is different from the current agent, append it to the agent
        # name as a sub-agent.  Skip this if it's a historical reload
        if (
            agent_name is not None
            and agent_name not in ["user", "meta"]
            and agent_name != self._current_agent
            and role == "assistant"
        ):
            agent_name = f"{self._current_agent}:{agent_name}"

        else:
            agent_name = self._current_agent

        # if the role is 'assistant_historical', change it to 'assistant'
        if role == "assistant_historical":
            role = "assistant"

        # Todo - find out why this is happening
        if message is None:
            ui.notify("add_to_chat_messsage: Message is None", type="warning")
            return

        if self.chatbox is None:
            if role == "user":
                self.chatbox = ChatTurn(
                    self.message_container,
                    role=role,
                    title=agent_name,
                    question=message,
                )
            else:
                self.chatbox = ChatTurn(
                    self.message_container, role=role, title=agent_name
                )
                await self.chatbox.append_chunk(message)
        else:
            await self.chatbox.append_chunk(message)

        # move spinner to the end if it exists
        if hasattr(self, "_spinner") and self._spinner.is_deleted is False:
            self._spinner.move(self.message_container)

    async def end_chat_turn(self, role: str, agent_name: str | None = None):
        """Marks the end of a portion of the conversation and stores it"""

        # user indicates a new message, assistant indicates a response
        # meta is neither and does not go in the session history
        assert role in ["user", "assistant", "meta"]

        if self.chatbox is None:
            return

        # note if we're seeing a response from a sub-agent
        if agent_name is not None and agent_name != self._current_agent:
            agent_name = f"{self._current_agent}:{agent_name}"
        else:
            agent_name = self._current_agent

        # if the role is 'user', this is a new message so setup a new turn
        if role == "user":
            self.record.new_turn(
                agent=agent_name,
                prompt=self.current_question,
                model=self._current_llm_model,
                temperature=self.llm_temperature,
                memory_messages=await self.ag.get_memory(),
            )

        # If we have a response, add the response to the current turn
        if role == "assistant":
            self.record.add_to_turn(
                agent_name=agent_name,
                response=self.chatbox.response,
                memory_messages=await self.ag.get_memory(),
            )

            await self.history_container.update_conversation(self.record)

        # # Update debug pane
        # debug_pane = self.query_one(DebugPane)
        # await debug_pane.update_status()

        # if we have a chatbox, end it
        self.chatbox = None

    async def ask_question(self, question: str):
        """Read the user's question and take action"""
        if question.lower() == "help":
            # await self.end_chat_turn(role="meta")
            await self.add_to_chat_message(
                role="assistant",
                message=(
                    "Available Commands:\n\n"
                    " - new: start a new session\n"
                    " - agents: show available agents\n"
                    " - agent <agent>: set the agent\n"
                    " - models: show available models\n"
                    " - model <model>: set the model\n"
                    " - temperature <temperature>: set the temperature\n"
                    " - summary: summarize the conversation\n"
                    " - stream [on|off]: turn stream tokens on or off\n"
                    " - dall-e <prompt>: generate an image from the prompt\n"
                ),
                agent_name="meta",
            )
            await self.end_chat_turn(role="meta")
            return

        # new
        if question.lower() == "new" or question == "new session":
            await self.end_chat_turn(role="meta")
            # if the session we're in is empty, don't start a new session
            if len(self.record.turns) == 0:
                await self.add_to_chat_message(
                    role="assistant",
                    message="You're already in a new session",
                    agent_name="meta",
                )
                await self.end_chat_turn(role="meta")
                return
            # Start a new session
            self.message_container.clear()
            self.record = await self.history_container.new_session()
            self._current_llm_model = self.mm.default_chat_model
            self.llm_temperature = self.mm.default_chat_temperature
            await self.set_agent_and_model(
                agent=self.default_agent,
                model=self.mm.default_chat_model,
                model_context=None,
            )
            return

        # agent
        if question.lower() == "agents" or question.lower() == "agent":
            await self.end_chat_turn(role="meta")
            await self.add_to_chat_message(
                role="assistant", message="Available agents\n\n", agent_name="meta"
            )

            for agent in self.chooseable_agents:
                if agent == self._current_agent:
                    await self.add_to_chat_message(
                        role="assistant",
                        message=f" - *{agent}* (current)\n",
                        agent_name="meta",
                    )
                else:
                    await self.add_to_chat_message(
                        role="assistant", message=f" - {agent}\n", agent_name="meta"
                    )
            await self.end_chat_turn(role="meta")
            return

        # set agent
        if question.startswith("agent"):
            await self.end_chat_turn(role="meta")
            # load the new agent
            agent = question.split(maxsplit=1)[1].strip()
            if agent not in self.agents:
                await self.add_to_chat_message(
                    role="assistant",
                    message=f"agent '{agent}' not found",
                    agent_name="meta",
                )
                await self.end_chat_turn(role="meta")
                return
            await self.set_agent_and_model(agent=agent)
            await self.add_to_chat_message(
                role="assistant",
                message=f"Agent set to {agent}\n\nModel set to {self.ag.model}",
                agent_name="meta",
            )
            await self.end_chat_turn(role="meta")
            return

        # available models
        if question == "models" or question == "model":
            await self.end_chat_turn(role="meta")
            await self.add_to_chat_message(
                role="assistant", message="Available Models:\n\n", agent_name="meta"
            )

            await self.add_to_chat_message(
                role="assistant", message="- LLM Models:\n", agent_name="meta"
            )
            for model in self.available_llm_models:
                await self.add_to_chat_message(
                    role="assistant",
                    message=f"   - {model}\n",
                    agent_name="meta",
                )
            await self.add_to_chat_message(
                role="assistant",
                message="- Image Models:\n",
                agent_name="meta",
            )
            for model in self.available_image_models:
                await self.add_to_chat_message(
                    role="assistant",
                    message=f"   - {model}\n",
                    agent_name="meta",
                )

            self.current_question = ""
            await self.end_chat_turn(role="meta")
            return

        # set model
        if question.startswith("model"):
            await self.end_chat_turn(role="meta")
            # get the model
            model = question.split(maxsplit=1)[1].strip()

            # check to see if the name is an llm or image model
            if model in self.available_llm_models:
                self._current_llm_model = model
                logger.debug(f"switching to llm model {model}")
                try:
                    await self.set_agent_and_model(model=model)
                except ValueError as e:
                    await self.add_to_chat_message(
                        role="assistant",
                        message=f"Error setting model: {e}",
                        agent_name="meta",
                    )

                else:
                    await self.add_to_chat_message(
                        role="assistant",
                        message=f"Model set to {self._current_llm_model}",
                        agent_name="meta",
                    )
                await self.end_chat_turn(role="meta")
                self.current_question = ""
                return
            elif model in self.available_image_models:
                self.image_model_name = model
                logger.debug(f"switching to image model {model}")
                self._reinitialize_image_model()
                await self.add_to_chat_message(
                    role="assistant",
                    message=f"Image model set to {self.image_model_name}",
                    agent_name="meta",
                )
                self.current_question = ""
                await self.end_chat_turn(role="meta")
                return
            else:
                await self.add_to_chat_message(
                    role="assistant",
                    message=f"Model '{model}' not found",
                    agent_name="meta",
                )
                self.current_question = ""
                self.end_chat_turn(role="meta")
                return

        # streaming
        if question.startswith("stream on"):
            await self.end_chat_turn(role="meta")
            self.ag.stream_tokens = True
            # if the agent doesn't support streaming, stream_tokens will be None
            if self.ag.stream_tokens is not None:
                # self.query_one(StatusBar).enable_stream_selector()
                # self.query_one(StatusBar).set_streaming(self.ag.stream_tokens)
                await self.add_to_chat_message(
                    role="assistant", message="Stream tokens are on", agent_name="meta"
                )
            else:
                # self.query_one(StatusBar).disable_stream_selector()
                await self.add_to_chat_message(
                    role="assistant",
                    message="Stream tokens are currently disabled",
                    agent_name="meta",
                )

            await self.end_chat_turn(role="meta")
            self.status_container.stream_switch.refresh()
            return
        if question.startswith("stream off"):
            await self.end_chat_turn(role="meta")
            self.ag.stream_tokens = False
            # if the agent doesn't support streaming, stream_tokens will be None
            if self.ag.stream_tokens is not None:
                # self.query_one(StatusBar).enable_stream_selector()
                # self.query_one(StatusBar).set_streaming(self.ag.stream_tokens)
                await self.add_to_chat_message(
                    role="assistant", message="Stream tokens are off", agent_name="meta"
                )
            else:
                # self.query_one(StatusBar).disable_stream_selector()
                await self.add_to_chat_message(
                    role="assistant",
                    message="Stream tokens are currently disabled",
                    agent_name="meta",
                )

            await self.end_chat_turn(role="meta")
            return
        if question == ("stream"):
            await self.end_chat_turn(role="meta")
            await self.add_to_chat_message(
                role="assistant",
                message=(
                    f"Stream tokens are " f"{'on' if self.ag.stream_tokens else 'off'}"
                ),
                agent_name="meta",
            )
            await self.end_chat_turn(role="meta")
            return

        # Normal user question
        self.current_question = question
        await self.end_chat_turn(role="user")
        try:
            await self.ag.ask(question)
        except Exception as e:
            logger.critical(f"Error running autogen: {e}")
            await self.add_to_chat_message(
                role="assistant",
                message=f"Error running autogen: {e}",
                agent_name="meta",
            )

        # Assistant is done
        await self.end_chat_turn(role="assistant")

    def load_agents(self) -> dict:
        """Read the agent definition files and load the agents"""
        if os.path.exists(DEFAULT_AGENT_FILE):
            extension = os.path.splitext(DEFAULT_AGENT_FILE)[1]
            with open(DEFAULT_AGENT_FILE) as f:
                if extension == ".json":
                    agents = json.load(f)
                elif extension == ".yaml":
                    agents = yaml.safe_load(f)
                else:
                    raise ValueError(
                        f"unknown extension {extension} for {DEFAULT_AGENT_FILE}"
                    )
        else:
            raise ValueError(f"no {DEFAULT_AGENT_FILE} file found")

        # if there is an EXTRA_AGENTS_FILE, load the agents from there
        if os.path.exists(EXTRA_AGENTS_FILE):
            extension = os.path.splitext(EXTRA_AGENTS_FILE)[1]
            with open(EXTRA_AGENTS_FILE, encoding="UTF-8") as f:
                if extension == ".json":
                    extra_agents = json.load(f)
                elif extension == ".yaml":
                    extra_agents = yaml.safe_load(f)
                else:
                    raise ValueError(
                        f"unknown extension {extension} for {EXTRA_AGENTS_FILE}"
                    )
            agents.update(extra_agents)
        return agents

    async def load_record(self, record):
        """Load a record into the chat, called from the history container"""
        self.record = record

        # if there are no turns in the record, it's a new session
        if len(self.record.turns) == 0:
            self.current_question = ""
            self.ag.clear_memory()
            await self.set_agent_and_model(
                agent=self.default_agent, model=self.mm.default_chat_model
            )
        else:
            # load the parameters from the last turn and reinitialize
            await self.set_agent_and_model(
                agent=self.record.turns[-1].agent,
                model=self.record.turns[-1].model,
                temperature=self.record.turns[-1].temperature,
                model_context=self.record.turns[-1].memory_messages,
            )
            self.current_question = self.record.turns[-1].prompt

        # clear the chatboxes from the chat container
        self.message_container.clear()

        # load the chat history from the record
        for turn in self.record.turns:
            await self.add_to_chat_message(role="user", message=turn.prompt)
            await self.end_chat_turn(role="meta")

            for response in turn.responses:
                await self.add_to_chat_message(
                    role="assistant_historical",
                    message=response["response"],
                    agent_name=response["agent"],
                )
                await self.end_chat_turn(role="meta")

    async def end_button_pressed(self):
        """Stop the running agent"""
        ui.notify("Will stop the agent on next turn", type="warning")
        self.ag.terminate()
        logger.debug("Agent stopped via End button")
        await self.end_chat_turn(role="assistant")

    async def esc_button_pressed(self):
        """Hard-terminate the running agent"""
        ui.notify("Terminating the agent", type="negative")
        self.ag.cancel()
        logger.debug("Agent terminated via ESC button")
        # cleanup
        if self._spinner.is_deleted is False:
            self._spinner.delete()
        await self.end_chat_turn(role="assistant")
        self.ui_is_busy = False

    async def set_agent_and_model(
        self,
        agent: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        model_context: dict | Literal["preserve"] | None = "preserve",
    ) -> None:
        """Change the agent and/or, updating the model to a compatible one if needed."""
        if not agent:
            agent = self._current_agent
        if agent not in self.agents:
            raise ValueError(f"agent '{agent}' not found")

        # determine the model to use; if the agent defines a model, use it
        # otherwise try to use the current one, lastly use the default

        compatible_models = self.mm.get_compatible_models(agent, self.agents)

        if not model:
            if self.agents[agent].get("model", None) is not None:
                model = self.agents[agent]["model"]
            elif self._current_llm_model in compatible_models:
                model = self._current_llm_model
            else:
                model = self.mm.default_chat_model

        if model not in compatible_models:
            raise ValueError(f"model '{model}' not compatible with agent '{agent}'")

        if temperature is None:
            temperature = self.llm_temperature

        if model_context == "preserve":
            model_context = await self.ag.get_memory()

        try:
            self.conversation = await self.ag.new_conversation(
                agent=agent, model_id=model, temperature=temperature
            )
        except Exception as e:
            logger.critical(f"Error setting agent and model: {e}")
            ui.notify(f"Error setting agent and model: {e}", type="warning")
            return

        self._current_agent = agent
        # _current_compatible_models is updated in the setter
        self.current_compatible_models = compatible_models
        if model != self.ag.model:
            logging.error(
                f"Setting model to {model} failed, model currently {self.ag.model}"
            )
        self._current_llm_model = self.ag.model
        self.llm_temperature = temperature

        # update the memory with the model context
        if model_context:
            await self.ag.update_memory(model_context)

        self.current_question = ""

        logger.info(f"Agent set to {agent}")
        logger.info(f"Model set to {self.ag.model}")


# if __name__ in {"__main__", "__mp_main__"}:
#     mchat_app = WebChatApp()
#     mchat_app.run(
#         port=8881,
#         title="MChat - Multi-Model Chat Framework",
#         favicon="static/favicon-32x32.png",
#         dark=True,
#     )
