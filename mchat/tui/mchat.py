import argparse
import asyncio
import json
import os
from typing import Callable, List, Literal, Tuple

import pyperclip
import yaml
from retry import retry
from textual import events, on, work
from textual.app import App, ComposeResult, Logger
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import Reactive
from textual.widgets import Footer, Header
from textual.worker import Worker, WorkerState
from textual_dominfo import DOMInfo

from config import settings
from mchat.llm import AutogenManager, LLMTools, ModelManager
from mchat.mchatweb import WebChatApp
from mchat.widgets.ChatTurn import ChatTurn
from mchat.widgets.DebugPane import DebugPane
from mchat.widgets.Dialog import Dialog
from mchat.widgets.FilePicker import FilePickerDialog
from mchat.widgets.History import HistoryContainer
from mchat.widgets.PromptInput import PromptInput
from mchat.widgets.StatusBar import StatusBar

DEFAULT_AGENT_FILE = "mchat/default_agents.yaml"
EXTRA_AGENTS_FILE = settings.get("extra_agents_file", None)


class StreamTokenCallback(object):
    """Callback handler that posts new tokens to the chatbox."""

    def __init__(self, app, *args, **kwargs):
        self.app = app

    # this method is automatically called by the Langchain callback system when a new
    # token is available
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Post the new token to the chatbox."""
        # if an agent is set, note it in the message
        agent_name = kwargs.get("agent", None)

        # if the token is empty, don't post it
        if token == "":
            return

        self.app.post_message(
            self.app.AddToChatMessage(
                role="assistant", message=token, agent_name=agent_name
            )
        )

        # if this is marked as a complete message, end the turn
        if "complete" in kwargs and kwargs["complete"]:
            self.app.post_message(
                self.app.EndChatTurn(role="assistant", agent_name=agent_name)
            )
        # await asyncio.sleep(0.05)


class ChatApp(App):
    CSS_PATH = "mchat.tcss"
    SCREENS = {"dialog": Dialog, "file_picker": FilePickerDialog}
    BINDINGS = [
        ("ctrl+r", "toggle_dark", "Toggle dark mode"),
        ("ctrl+g", "toggle_debug", "Toggle debug mode"),
        (
            "ctrl+q",
            "confirm_y_n('[bold]Quit?[/bold] Y/N', 'my_quit', 'dialog_close', "
            "'[Quit]')",
            "Quit",
        ),
        (
            "ctrl+o",
            "select_file('PDF File to Open', 'show_file', 'dialog_close')",
            "Open File",
        ),
        # ("ctrl+c", "my_quit", "quit"),
        # ("ctrl+t", "toggle_css_tooltip", "CSS tooltip"),
    ]

    # Toggles debug pane on/off (default is off)
    _show_debug = Reactive(False)

    # placeholder for the current question
    _current_question = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # current debug log
        self.debug_log = ""

        # parse arguments
        self.parse_args_and_initialize()

        # current active widget for storing a conversation turn
        self.chatbox = None

        # load standard agents
        if os.path.exists(DEFAULT_AGENT_FILE):
            extension = os.path.splitext(DEFAULT_AGENT_FILE)[1]
            with open(DEFAULT_AGENT_FILE) as f:
                if extension == ".json":
                    self.agents = json.load(f)
                elif extension == ".yaml":
                    self.agents = yaml.safe_load(f)
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
            self.agents.update(extra_agents)

        # Get an object to manage the AI models
        self.mm = ModelManager()

        # Get an object to manage autogen
        self.ag = AutogenManager(
            message_callback=StreamTokenCallback(self).on_llm_new_token,
            agents=self.agents,
        )

        # load the llm models from settings - name: {key: value,...}

        self.available_llm_models = self.mm.available_chat_models
        self.available_image_models = self.mm.available_image_models

        self.llm_model_name = self.mm.default_chat_model
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

        self.ready = False
        # The conversation is initialized in the on_mount method

    def _reinitialize_image_model(self):
        """re-initialize the image model."""
        self.image_model = self.mm.open_model(self.image_model_name, model_type="image")

    def parse_args_and_initialize(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-v", "--verbose", help="Increase verbosity", action="store_true"
        )
        args, unknown = parser.parse_known_args()

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        with Horizontal():
            yield DebugPane()
            yield HistoryContainer(new_label="New Session")
            with Vertical(id="right-pane"):
                self.chat_container = VerticalScroll(id="chat-container")
                yield self.chat_container
                yield PromptInput()
                yield StatusBar()
        yield Footer()

    async def on_mount(self) -> None:
        self.title = "mchat - Multi-Model Chatbot"

        # Initialize the conversation
        self.default_agent = getattr(settings, "defaults.agent", "default")

        # load the agents and select the default
        self.query_one(StatusBar).load_agents(
            [
                (agent, agent)
                for agent, data in self.agents.items()
                if data.get("chooseable", True)
            ],
            value=self.default_agent,
        )

        await self.set_agent_and_model(
            agent=self.default_agent,
            model=self.llm_model_name,
            temperature=self.llm_temperature,
        )

        # set focus to the input box
        input = self.query_one(PromptInput)
        input.focus()

        # Load the active conversation record
        self.record = self.query_one(HistoryContainer).active_record

        self.ready = True

    async def on_ready(self) -> None:
        """Called  when the DOM is ready."""
        input = self.query_one(PromptInput)
        input.focus()

        # populate the debug pane
        debug_pane = self.query_one(DebugPane)
        await debug_pane.add_entry("model", "LLM Model", lambda: self.llm_model_name)
        await debug_pane.add_entry(
            "imagemodel", "Image Model", lambda: self.image_model_name
        )
        await debug_pane.add_entry("temp", "Temperature", lambda: self.llm_temperature)
        await debug_pane.add_entry("agent", "Agent", lambda: self.current_agent)
        await debug_pane.add_entry(
            "question", "Question", lambda: self._current_question, collapsed=True
        )

        await debug_pane.add_entry(
            "description",
            "Description",
            lambda: self.app.ag.description,
            collapsed=True,
        )

        await debug_pane.add_entry(
            "prompt", "Prompt", lambda: self.app.ag.prompt, collapsed=True
        )

        await debug_pane.add_entry(
            "memref",
            "Memory Reference",
            self.ag.get_memory,
            collapsed=True,
        )

        # debug_pane.add_entry(
        #     "summary_buffer",
        #     "Summary Buffer",
        #     lambda: self.memory.moving_summary_buffer,
        #     collapsed=True,
        # )
        await debug_pane.add_entry("log", "Debug Log", lambda: self.debug_log)

        # monkey patch the debug logger so we can watch app logs in the debug pane
        app_debug_logger = self.log.debug

        def update_log(msg):
            self.debug_log += f"{msg}\n"
            if self.app.is_mounted(debug_pane):
                asyncio.create_task(debug_pane.update_entry("log"))
                # loop = asyncio.get_event_loop()
                # task = asyncio.create_task(debug_pane.update_entry("log"))
                # loop.run_until_complete(task)
                # debug_pane.update_entry("log")

        def debug(self, msg, *args, **kwargs):
            app_debug_logger(msg, *args, **kwargs)
            if isinstance(msg, str):
                update_log(msg)

        Logger.debug = debug

        # update the logger for the model manager and autogen manager
        self.mm.logger = self.log.debug
        self.ag.logger = self.log.debug

    @on(ChatTurn.ChatTurnClicked)
    def click_chat_turn(self, event: events) -> None:
        chatturn = event.widget

        if event.local_text:
            # copy contents of chatbox to clipboard
            pyperclip.copy(event.local_text)
        else:
            # copy contents of chatbox to clipboard
            pyperclip.copy(chatturn.message)

        self.notify("Copied to clipboard", timeout=1)

    @on(PromptInput.Submitted)
    def submit_question(self, event: events) -> None:
        input = self.query_one(PromptInput)

        # don't post "new" or "new session" commands to the chatbox
        if event.value.lower() != "new" and event.value.lower() != "new session":
            self.post_message(self.AddToChatMessage(role="user", message=event.value))

        # clear the input box
        input.value = ""

        # ask_question is a work function, so it will be run in a separate thread
        self.ask_question(event.value)

    @on(StatusBar.AgentChangedMessage)
    async def change_agent_handler(
        self, message: StatusBar.AgentChangedMessage
    ) -> None:
        agent = message.agent
        if agent and agent != self.current_agent:
            await self.set_agent_and_model(agent=agent, model_context="preserve")

            self.post_message(
                self.AddToChatMessage(
                    role="assistant",
                    message=f"Agent set to {agent}\n\nModel set to {self.ag.model}",
                    agent_name="meta",
                )
            )
            self.post_message(self.EndChatTurn(role="meta"))

    @on(StatusBar.ModelChangedMessage)
    async def change_model(self, message: StatusBar.ModelChangedMessage) -> None:
        """Change the model to the on selected in the status bar."""
        model = message.model
        if model and model != self.llm_model_name:
            await self.set_agent_and_model(
                model=message.model, model_context="preserve"
            )

            self.post_message(
                self.AddToChatMessage(
                    role="assistant",
                    message=f"Model set to {self.ag.model}",
                    agent_name="meta",
                )
            )
            self.post_message(self.EndChatTurn(role="meta"))

    @on(StatusBar.EndButtonPressedMessage)
    async def end_button_pressed(
        self, message: StatusBar.EndButtonPressedMessage
    ) -> None:
        """End the current conversation turn."""
        self.ag.terminate()
        self.log.debug("End button pressed")
        self.post_message(self.EndChatTurn(role="assistant"))
        self.notify("Conversation will end shortly", timeout=2)

    @on(StatusBar.EscapeButtonPressedMessage)
    async def escape_button_pressed(
        self, message: StatusBar.EscapeButtonPressedMessage
    ) -> None:
        """End the current team execution immediately."""
        self.ag.cancel()
        self.log.debug("Escape button pressed")
        self.post_message(self.EndChatTurn(role="assistant"))
        self.notify("Conversation cancelled", severity="error", timeout=2)

    async def set_agent_and_model(
        self,
        agent: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        model_context: dict | Literal["preserve"] | None = None,
        update_ui: bool = False,
    ) -> None:
        """Change the agent and/or, updating the model to a compatible one if needed."""
        if not agent:
            agent = self.current_agent
        if agent not in self.agents:
            raise ValueError(f"agent '{agent}' not found")

        # determine the model to use; if the agent defines a model, use it
        # otherwise try to use the current one, lastly use the default

        compatible_models = self.mm.get_compatible_models(agent, self.agents)

        if not model:
            if self.agents[agent].get("model", None) is not None:
                model = self.agents[agent]["model"]
            elif self.llm_model_name in compatible_models:
                model = self.llm_model_name
            else:
                model = self.mm.default_chat_model

        if model not in compatible_models:
            raise ValueError(f"model '{model}' not compatible with agent '{agent}'")

        if temperature is None:
            temperature = self.llm_temperature

        if model_context == "preserve":
            model_context = await self.ag.get_memory()

        self.conversation = await self.ag.new_conversation(
            agent=agent, model_id=model, temperature=temperature
        )

        # if it is a new agent, load compatable models into statusbar
        if agent != getattr(self, "current_agent", None):
            self.query_one(StatusBar).load_models(
                [(model, model) for model in compatible_models],
                value=model,
            )

        # TODO rename llm_model_name to current_model
        self.current_agent = agent
        self.llm_model_name = model
        self.llm_temperature = temperature

        # update the memory with the model context
        if model_context:
            await self.ag.update_memory(model_context)

        self._current_question = ""

        # if setting streaming is supported, enable the streaming selector
        if self.ag.stream_tokens is not None:
            self.query_one(StatusBar).enable_stream_selector()
            self.query_one(StatusBar).set_streaming(self.ag.stream_tokens)
        else:
            self.query_one(StatusBar).disable_stream_selector()

        # update the agent in the status bar if needed
        # (e.g. if the agent was set by command)
        if update_ui and self.query_one(StatusBar).agent != agent:
            self.query_one(StatusBar).agent = agent

        # show the current model if not already set
        if update_ui and self.query_one(StatusBar).model != model:
            self.query_one(StatusBar).model = model

        self.log.debug(f"Agent set to {agent}")
        self.log.debug(f"Model set to {self.ag.model}")

        debug_pane = self.query_one(DebugPane)
        # debug_pane.update_status()
        # debug_pane.update_entry(
        #     "summary_buffer",
        #     lambda: self.memory.moving_summary_buffer,
        # )

        await debug_pane.update_status()

    @on(StatusBar.StreamingChangedMessage)
    def change_streaming_handler(
        self, message: StatusBar.StreamingChangedMessage
    ) -> None:
        self.ag.stream_tokens = message.streaming

    def on_key(self, event: events.Key) -> None:
        """Write Key events to log."""
        pass
        # text_log = self.query_one(TextLog)
        # text_log.write(event)
        # self.post_message(self.AddToChatMessperage("keypress"))

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.theme = (
            "textual-dark" if self.theme == "solarized-light" else "solarized-light"
        )

    def action_toggle_debug(self) -> None:
        """An action to toggle debug mode."""
        self._show_debug = not self._show_debug

    def action_confirm_y_n(
        self, message: str, confirm_action: str, noconfirm_action: str, title: str = ""
    ) -> None:
        """Build Yes/No Modal Dialog and process callbacks."""
        self.app.push_screen("dialog")
        dialog = self.query_one("#modal_dialog", Dialog)
        dialog.set_message(message)
        dialog.confirm_action = confirm_action
        dialog.noconfirm_action = noconfirm_action

    async def action_select_file(
        self,
        message: str = "Select a file to open",
        confirm_action: str = "close_dialog",
        noconfirm_action: str = "close_dialog",
    ) -> None:
        """Open the file picker dialog"""
        self.push_screen("file_picker")
        file_picker = self.query_one("#file_picker", FilePickerDialog)
        file_picker.message = message
        file_picker.confirm_action = confirm_action
        file_picker.noconfirm_action = noconfirm_action
        file_picker.allowed_extensions = [".pdf"]

    def action_my_quit(self) -> None:
        """Quit the app."""
        self.log.debug("Quitting")
        self.app.exit()

    def action_show_file(self) -> None:
        """Show the selected file"""
        file_picker = self.query_one("#file_picker", FilePickerDialog)
        self.app.log.debug(
            f"File {file_picker.selected_path} was selected by the file picker"
        )

    def action_toggle_css_tooltip(self) -> None:
        """Toggle the CSS tooltip."""
        DOMInfo.attach_to(self)

    def watch__show_debug(self, show_debug: bool) -> None:
        """When __show_debug changes, toggle the class the debug widget."""
        self.app.set_class(show_debug, "-show-debug")

    @work(exclusive=True)
    async def ask_question(self, question: str):
        """Read the user's question and take action.  Textual work function."""

        # Figure out what the user wants to do
        # intent = await self.cm.aget_intent(question, memory=self.memory)

        # intent = self.cm.get_intent(question, memory=self.memory)
        # self.app.log.debug(f"User intent: {intent}")

        # await self._determine_user_intent(question, memory=self.memory)

        # scroll the chat container to the bottom
        self.scroll_to_end()

        # if the question is 'help', show the help message
        if question.lower() == "help":
            self.post_message(self.EndChatTurn(role="meta"))
            self.post_message(
                self.AddToChatMessage(
                    role="assistant",
                    message=(
                        "Available Commands:\n"
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
                )
            )
            self.post_message(self.EndChatTurn(role="meta"))
            return

        # if the question is 'new' or 'new session" start a new session
        if question.lower() == "new" or question == "new session":
            self.post_message(self.EndChatTurn(role="meta"))
            # if the session we're in is empty, don't start a new session
            if len(self.record.turns) == 0:
                self.post_message(
                    self.AddToChatMessage(
                        role="assistant",
                        message="You're already in a new session",
                        agent_name="meta",
                    )
                )
                self.post_message(self.EndChatTurn(role="meta"))
                return

            # clear the chatboxes from the chat container
            self.chat_container.remove_children()

            # start a new history session
            history = self.query_one(HistoryContainer)
            self.record = await history.new_session()
            self.llm_model_name = self.mm.default_chat_model
            self.llm_temperature = self.mm.default_chat_temperature
            await self.set_agent_and_model(agent=self.default_agent, update_ui=True)
            return

        # if the question is either 'agent', or 'agents' show the available agents
        if question == "agents" or question == "agent":
            self.post_message(self.EndChatTurn(role="meta"))
            self.post_message(
                self.AddToChatMessage(
                    role="assistant", message="Available agents\n", agent_name="meta"
                )
            )
            for agent in [
                agent_name
                for agent_name, val in self.agents.items()
                if val.get("chooseable", True)
            ]:
                if agent == self.current_agent:
                    self.post_message(
                        self.AddToChatMessage(
                            role="assistant",
                            message=f" - *{agent}* (current)\n",
                            agent_name="meta",
                        ),
                    )
                else:
                    self.post_message(
                        self.AddToChatMessage(
                            role="assistant", message=f" - {agent}\n", agent_name="meta"
                        ),
                    )
            self.post_message(self.EndChatTurn(role="meta"))
            return

        if question.startswith("agent"):
            self.post_message(self.EndChatTurn(role="meta"))
            # load the new agent
            agent = question.split(maxsplit=1)[1].strip()
            if agent not in self.agents:
                self.post_message(
                    self.AddToChatMessage(
                        role="assistant",
                        message=f"agent '{agent}' not found",
                        agent_name="meta",
                    )
                )
                self.post_message(self.EndChatTurn(role="meta"))
                return
            await self.set_agent_and_model(
                agent=agent, model_context="preserve", update_ui=True
            )
            self.post_message(
                self.AddToChatMessage(
                    role="assistant",
                    message=f"Agent set to {agent}\n\nModel set to {self.ag.model}",
                    agent_name="meta",
                )
            )
            self.post_message(self.EndChatTurn(role="meta"))
            return

        # if the question is 'models', or 'model', show available models
        if question == "models" or question == "model":
            self.post_message(self.EndChatTurn(role="meta"))
            self.post_message(
                self.AddToChatMessage(
                    role="assistant", message="Available Models:\n", agent_name="meta"
                )
            )
            self.post_message(
                self.AddToChatMessage(
                    role="assistant", message="- LLM Models:\n", agent_name="meta"
                )
            )
            for model in self.available_llm_models:
                self.post_message(
                    self.AddToChatMessage(
                        role="assistant",
                        message=f"   - {model}\n",
                        agent_name="meta",
                    )
                )
            self.post_message(
                self.AddToChatMessage(
                    role="assistant",
                    message="- Image Models:\n",
                    agent_name="meta",
                )
            )
            for model in self.available_image_models:
                (
                    self.post_message(
                        self.AddToChatMessage(
                            role="assistant",
                            message=f"   - {model}\n",
                            agent_name="meta",
                        )
                    ),
                )
            self._current_question = ""
            self.post_message(self.EndChatTurn(role="meta"))
            return

        # if the question starts with 'model', set the model
        if question.startswith("model"):
            self.post_message(self.EndChatTurn(role="meta"))
            # get the model
            model = question.split(maxsplit=1)[1].strip()

            # check to see if the name is an llm or image model
            if model in self.available_llm_models:
                self.llm_model_name = model
                self.log.debug(f"switching to llm model {model}")
                try:
                    await self.set_agent_and_model(
                        model=model, model_context="preserve", update_ui=True
                    )
                except ValueError as e:
                    self.post_message(
                        self.AddToChatMessage(
                            role="assistant",
                            message=f"Error setting model: {e}",
                            agent_name="meta",
                        )
                    )
                else:
                    self.post_message(
                        self.AddToChatMessage(
                            role="assistant",
                            message=f"Model set to {self.llm_model_name}",
                            agent_name="meta",
                        )
                    )
                self.post_message(self.EndChatTurn(role="meta"))
                self._current_question = ""
                return
            elif model in self.available_image_models:
                self.image_model_name = model
                self.log.debug(f"switching to image model {model}")
                self._reinitialize_image_model()
                self.post_message(
                    self.AddToChatMessage(
                        role="assistant",
                        message=f"Image model set to {self.image_model_name}",
                        agent_name="meta",
                    )
                )
                self._current_question = ""
                self.post_message(self.EndChatTurn(role="meta"))
                return
            else:
                self.post_message(
                    self.AddToChatMessage(
                        role="assistant",
                        message=f"Model '{model}' not found",
                        agent_name="meta",
                    )
                )
                self._current_question = ""
                self.post_message(self.EndChatTurn(role="meta"))
                return

        # if the question starts with 'summary', summarize the conversation
        if question.startswith("summary"):
            self.post_message(self.EndChatTurn(role="meta"))
            self.post_message(
                self.AddToChatMessage(
                    role="assistant", message="Summarizing...", agent_name="meta"
                )
            )
            summary = await LLMTools.get_conversation_summary(self.record)
            self._current_question = ""
            self.post_message(self.AddToChatMessage(role="assistant", message=summary))
            self.post_message(self.EndChatTurn(role="meta"))
            return

        # if the question starts with 'temperature', set the temperature
        if question.startswith("temperature"):
            self.post_message(self.EndChatTurn(role="meta"))
            # get the temperature
            self.llm_temperature = float(question.split(maxsplit=1)[1].strip())
            # post the summary
            self.post_message(
                self.AddToChatMessage(
                    role="assistant",
                    message=f"Temperature set to {self.llm_temperature}",
                    agent_name="meta",
                )
            )
            self._current_question = question
            await self.set_agent_and_model(
                temperature=self.llm_temperature, model_context="preserve"
            )
            self.post_message(self.EndChatTurn(role="assistant", agent_name="meta"))
            return

        # if the question starts with 'stream', set the stream tokens
        if question.startswith("stream on"):
            self.post_message(self.EndChatTurn(role="meta"))

            self.ag.stream_tokens = True
            # if the agent doesn't support streaming, stream_tokens will be None
            if self.ag.stream_tokens is not None:
                self.query_one(StatusBar).enable_stream_selector()
                self.query_one(StatusBar).set_streaming(self.ag.stream_tokens)
                self.post_message(
                    self.AddToChatMessage(
                        role="assistant", message="Stream tokens are on"
                    )
                )
            else:
                self.query_one(StatusBar).disable_stream_selector()
                self.post_message(
                    self.AddToChatMessage(
                        role="assistant", message="Stream tokens are currently disabled"
                    )
                )

            self.post_message(self.EndChatTurn(role="meta"))
            return
        if question.startswith("stream off"):
            self.post_message(self.EndChatTurn(role="meta"))
            self.ag.stream_tokens = False
            # if the agent doesn't support streaming, stream_tokens will be None
            if self.ag.stream_tokens is not None:
                self.query_one(StatusBar).enable_stream_selector()
                self.query_one(StatusBar).set_streaming(self.ag.stream_tokens)
                self.post_message(
                    self.AddToChatMessage(
                        role="assistant", message="Stream tokens are off"
                    )
                )
            else:
                self.query_one(StatusBar).disable_stream_selector()
                self.post_message(
                    self.AddToChatMessage(
                        role="assistant", message="Stream tokens are currently disabled"
                    )
                )

            self.post_message(self.EndChatTurn(role="meta"))
            return
        if question == ("stream"):
            self.post_message(self.EndChatTurn(role="meta"))
            self.post_message(
                self.AddToChatMessage(
                    role="assistant",
                    message=(
                        f"Stream tokens are "
                        f"{'on' if self.ag.stream_tokens else 'off'}"
                    ),
                )
            )
            self.post_message(self.EndChatTurn(role="meta"))
            return

        # if question starts with 'dall-e ' pass to Dall-e
        if question.startswith("dall-e "):
            self.post_message(self.EndChatTurn(role="user"))

            self._current_question = question
            question = question[7:]
            self.post_message(
                self.AddToChatMessage(
                    role="assistant", message="Generating...", agent_name="dall-e"
                )
            )
            self.post_message(self.EndChatTurn(role="meta"))
            try:
                out = await self.image_model.arun(question)
            except Exception as e:
                self.post_message(
                    self.AddToChatMessage(
                        role="assistant", message=f"Error generating image: {e}"
                    )
                )
                self.post_message(self.EndChatTurn(role="meta"))
                return
            self.post_message(
                self.AddToChatMessage(
                    role="assistant", message=out, agent_name="dall-e"
                )
            )
            self.post_message(self.EndChatTurn(role="assistant", agent_name="dall-e"))
            return

        # Just a normal question at this point
        self.post_message(self.EndChatTurn(role="user"))
        self._current_question = question

        # ask the question and wait for a response, the routine is responsible for
        # posting any tokens to the chatbox via the callback function passed when the
        # object was created

        # change instructions to mark loading
        instructions = self.query_one("#instructions")
        instructions.loading = True
        self.query_one(StatusBar).enable_end_button()
        self.query_one(StatusBar).enable_escape_button()

        try:
            await self.ag.ask(question)
        except Exception as e:
            self.post_message(
                self.AddToChatMessage(
                    role="assistant",
                    message=f"Error running autogen: {e}",
                    agent_name="meta",
                )
            )
        instructions.loading = False
        self.query_one(StatusBar).disable_end_button()
        self.query_one(StatusBar).disable_escape_button()

        # Done with response; clear the chatbox
        self.scroll_to_end()
        self.post_message(self.EndChatTurn(role="assistant"))

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Called when the worker state changes."""

        if event.state == WorkerState.CANCELLED:
            self.query_one("#instructions").loading = False
            self.query_one(StatusBar).disable_end_button()
            self.query_one(StatusBar).disable_escape_button()
            self.scroll_to_end()
        self.log(event)

    def run(self, *args, **kwargs):
        """Run the app."""

        try:
            super().run(*args, **kwargs)
        except asyncio.exceptions.CancelledError:
            self.log.debug("Markdown crashed\n{}", self.chatbox.markdown)

    def scroll_to_end(self) -> None:
        if self.chat_container is not None:
            self.chat_container.refresh()
            self.chat_container.scroll_end(animate=False)

    # Custom Message Handlers - these are in two parts:
    # 1. A Message class that is used to send the message and embed data
    # 2. A function that is decorated with @on(MessageClass) that will be called when
    #    the message is received

    class AddToChatMessage(Message):
        def __init__(
            self, role: str, message: str, agent_name: str | None = None
        ) -> None:
            assert role in ["user", "assistant", "assistant_historical"]
            self.role = role
            self.message = message
            self.agent_name = agent_name
            super().__init__()

    @on(AddToChatMessage)
    async def add_to_chat_message(self, chat_token: AddToChatMessage) -> None:
        chunk = chat_token.message
        role = chat_token.role
        agent_name = chat_token.agent_name

        # if the agent_name is different from the current agent, append it to the agent
        # name as a sub-agent.  Skip this if it's a historical reload
        if (
            agent_name is not None
            and agent_name not in ["user", "meta"]
            and agent_name != self.current_agent
            and role == "assistant"
        ):
            agent_name = f"{self.current_agent}:{agent_name}"

        # if the role is 'assistant_historical', change it to 'assistant'
        if role == "assistant_historical":
            role = "assistant"

        # Todo - find out why this is happening
        if chunk is None:
            return

        # if the Chatbox is not mounted, it is in the process of being deleted, so
        # ignore any messages headed for it
        if self.chatbox is not None and self.chatbox.is_mounted is False:
            return

        # Create a ChatTurn widget if we don't have one and mount it in the container
        # make sure to scroll to the bottom
        if self.chatbox is None:
            self.chatbox = ChatTurn(role=role, title=agent_name)
            await self.chat_container.mount(self.chatbox)
            # Attach the DOM inspector to the tooltip
            self.chat_container.scroll_end(animate=False)

        await self.chatbox.append_chunk(chunk)

        # if we're not near the bottom, scroll to the bottom
        scroll_y = self.chat_container.scroll_y
        max_scroll_y = self.chat_container.max_scroll_y
        if scroll_y in range(max_scroll_y - 3, max_scroll_y + 1):
            self.chat_container.scroll_end(animate=False)

    class EndChatTurn(Message):
        def __init__(self, role: str, agent_name: str | None = None) -> None:
            assert role in ["user", "assistant", "meta"]
            self.role = role
            self.agent_name = agent_name
            super().__init__()

    @on(EndChatTurn)
    async def end_chat_turn(self, event: EndChatTurn) -> None:
        """close the current chat turn"""

        # some situations can cause EndChatTurn to be successively called, so there
        # may not be an existing chatbox on the second call
        if self.chatbox is None:
            self.log("Received EndChatTurn message with no active chatbox")
            return

        if event.agent_name is not None and event.agent_name != self.current_agent:
            agent_name = f"{self.current_agent}:{event.agent_name}"
        else:
            agent_name = self.current_agent

        # if the role is 'user', setup a new turn
        if event.role == "user":
            self.record.new_turn(
                agent=agent_name,
                prompt=self._current_question,
                model=self.llm_model_name,
                temperature=self.llm_temperature,
                memory_messages=await self.ag.get_memory(),
            )

        # If we have a response, add the response to the current turn
        if event.role == "assistant":
            self.record.add_to_turn(
                agent_name=agent_name,
                response=self.chatbox.message,
                memory_messages=await self.ag.get_memory(),
            )

            history = self.query_one(HistoryContainer)
            await history.update_conversation(self.record)

        # Update debug pane
        debug_pane = self.query_one(DebugPane)
        await debug_pane.update_status()

        # if we have a chatbox, close it.
        self.chatbox = None

    @on(HistoryContainer.HistorySessionClicked)
    async def _on_history_session_clicked(
        self, event: HistoryContainer.HistorySessionClicked
    ) -> None:
        """Restore the selected session"""

        event.stop()

        # if the record is the same as the current record, do nothing
        if self.record == event.record:
            return

        self.record = event.record

        # if there are no turns in the record, it's a new session
        if len(self.record.turns) == 0:
            self._current_question = ""
            self.ag.clear_memory()
            await self.set_agent_and_model(
                agent=self.default_agent,
                model=self.mm.default_chat_model,
                update_ui=True,
            )
        else:
            # load the parameters from the last turn and reinitialize
            # self.current_agent = self.record.turns[-1].agent
            # self.llm_model_name = self.record.turns[-1].model
            # self.llm_temperature = self.record.turns[-1].temperature
            await self.set_agent_and_model(
                agent=self.record.turns[-1].agent,
                model=self.record.turns[-1].model,
                temperature=self.record.turns[-1].temperature,
                model_context=self.record.turns[-1].memory_messages,
                update_ui=True,
            )
            self._current_question = self.record.turns[-1].prompt

            # await self._reinitialize_llm_model(self.record.turns[-1].memory_messages)

        # clear the chatboxes from the chat container
        self.chat_container.remove_children()

        # load the chat history from the record
        self.chat_container.visible = False
        for turn in self.record.turns:
            self.post_message(self.AddToChatMessage(role="user", message=turn.prompt))
            self.post_message(self.EndChatTurn(role="meta"))
            for response in turn.responses:
                self.post_message(
                    self.AddToChatMessage(
                        role="assistant_historical",
                        message=response["response"],
                        agent_name=response["agent"],
                    )
                )
                self.post_message(self.EndChatTurn(role="meta"))
        self.scroll_to_end()
        self.chat_container.visible = True


def run():
    """Run the app"""
    app.run()


app = WebChatApp()
# app = ChatApp()

if __name__ in {"__main__", "__mp_main__"}:
    run()
