import os
import asyncio
import argparse
import pyperclip

from config import settings

from typing import Any, Dict, List, Optional, Callable

from retry import retry

from textual.app import App, ComposeResult, Logger
from textual.widgets import Header, Footer
from textual.containers import VerticalScroll, Vertical, Horizontal
from textual.reactive import Reactive
from textual.css.query import NoMatches
from textual import on, work
from textual import events
from textual.message import Message
from textual.worker import Worker
from textual_dominfo import DOMInfo


from mchat.widgets.ChatTurn import ChatTurn
from mchat.widgets.DebugPane import DebugPane
from mchat.widgets.PromptInput import PromptInput
from mchat.widgets.History import HistoryContainer
from mchat.widgets.Dialog import Dialog
from mchat.widgets.FilePicker import FilePickerDialog
from mchat.llm import AutogenManager, ModelManager

DEFAULT_PERSONA_FILE = "mchat/default_personas.json"
EXTRA_PERSONA_FILE = "extra_personas.json"


class StreamTokenCallback(object):
    """Callback handler that posts new tokens to the chatbox."""

    def __init__(self, app, *args, **kwargs):
        self.app = app

    # this method is automatically called by the Langchain callback system when a new
    # token is available
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.app.post_message(
            self.app.AddToChatMessage(role="assistant", message=token)
        )
        # await asyncio.sleep(0.05)


class ChatApp(App):
    CSS_PATH = "mchat.tcss"
    BINDINGS = [
        ("ctrl+r", "toggle_dark", "Toggle dark mode"),
        ("ctrl+g", "toggle_debug", "Toggle debug mode"),
        (
            "ctrl+q",
            "confirm_y_n('[bold]Quit?[/bold] Y/N', 'my_quit', 'dialog_close', '[Quit]')",
            "Quit",
        ),
        (
            "ctrl+o",
            "select_file('PDF File to Open', 'show_file', 'dialog_close')",
            "Open File",
        ),
        ("ctrl+c", "my_quit", "quit"),
        ("ctrl+t", "toggle_css_tooltip", "CSS tooltip"),
    ]

    # Toggles debug pane on/off (default is off)
    _show_debug = Reactive(False)

    # placeholder for the current question
    _current_question = Reactive("")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # current debug log
        self.debug_log = ""

        # parse arguments
        self.parse_args_and_initialize()

        # current active widget for storing a conversation turn
        self.chatbox = None

        # load standard personas
        if os.path.exists(DEFAULT_PERSONA_FILE):
            import json

            with open(DEFAULT_PERSONA_FILE) as f:
                self.personas = json.load(f)
        else:
            raise ValueError("no default_personas.json file found")

        # if there is an EXTRA_PERSONA_FILE, load the personas from there
        if os.path.exists(EXTRA_PERSONA_FILE):
            import json

            with open(EXTRA_PERSONA_FILE) as f:
                extra_personas = json.load(f)
            self.personas.update(extra_personas)

        # Get an object to manage the AI models
        self.mm = ModelManager()

        # Get an object to manage autogen
        self.ag = AutogenManager(
            message_callback=StreamTokenCallback(self).on_llm_new_token,
            personas=self.personas,
        )

        # load the llm models from settings - name: {key: value,...}

        self.available_llm_models = self.mm.available_chat_models
        self.available_image_models = self.mm.available_image_models

        self.llm_model_name = self.mm.default_chat_model
        self.llm_temperature = self.mm.default_chat_temperature

        # Initialize the image model
        self.default_image_model = settings.get("default_image_model", None)
        if self.mm.default_image_model is not None:
            self.image_model_name = self.mm.default_image_model
            self.image_model = self.mm.open_model(
                self.image_model_name, model_type="image"
            )

        # Initialize the conversation
        self.default_persona = getattr(settings, "default_persona", "default")
        self.set_persona(
            self.default_persona, self.llm_model_name, self.llm_temperature
        )

    def _reinitialize_llm_model(self, messages: List[str] = []):
        """re-initialize the language model."""

        self.conversation = self.ag.new_conversation(
            self.current_persona,
            model_id=self.llm_model_name,
            temperature=self.llm_temperature,
        )

        # if there are messages, we're restoring a historical session, create new
        # memory and reinitialize the conversation
        if len(messages) > 0:
            self.ag.update_memory(messages)

        debug_pane = self.query_one(DebugPane)

        # debug_pane.update_entry(
        #     "history", lambda: self.memory.load_memory_variables({})["history"]
        # )
        # debug_pane.update_entry(
        #     "summary_buffer",
        #     lambda: self.memory.moving_summary_buffer,
        # )

        debug_pane.update_status()

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
        yield Footer()
        yield Dialog(id="modal_dialog")
        yield FilePickerDialog(id="file_picker")

    def on_mount(self) -> None:
        self.title = "mchat - Multi-Model Chatbot"

        # set focus to the input box
        input = self.query_one(PromptInput)
        input.focus()

        # Load the active conversation record
        self.record = self.query_one(HistoryContainer).active_record

    def on_ready(self) -> None:
        """Called  when the DOM is ready."""
        input = self.query_one(PromptInput)
        input.focus()

        # populate the debug pane
        debug_pane = self.query_one(DebugPane)
        debug_pane.add_entry("model", "LLM Model", lambda: self.llm_model_name)
        debug_pane.add_entry("imagemodel", "Image Model", lambda: self.image_model_name)
        debug_pane.add_entry("temp", "Temperature", lambda: self.llm_temperature)
        debug_pane.add_entry("persona", "Persona", lambda: self.current_persona)
        debug_pane.add_entry(
            "question", "Question", lambda: self._current_question, collapsed=True
        )
        debug_pane.add_entry(
            "prompt", "Prompt", lambda: self.app.ag.prompt, collapsed=True
        )
        # debug_pane.add_entry(
        #     "history",
        #     "History",
        #     lambda: self.memory.load_memory_variables({})["history"],
        #     collapsed=True,
        # )
        debug_pane.add_entry(
            # "memref", "Memory Reference", lambda: self.memory, collapsed=True
            "memref",
            "Memory Reference",
            lambda: self.ag.memory,
            collapsed=True,
        )
        # debug_pane.add_entry(
        #     "summary_buffer",
        #     "Summary Buffer",
        #     lambda: self.memory.moving_summary_buffer,
        #     collapsed=True,
        # )
        debug_pane.add_entry("log", "Debug Log", lambda: self.debug_log)

        # monkey patch the debug logger so we can watch app logs in the debug pane
        app_debug_logger = self.log.debug

        def update_log(msg):
            self.debug_log += f"{msg}\n"
            if self.app.is_mounted(debug_pane):
                debug_pane.update_entry("log")

        def debug(self, msg, *args, **kwargs):
            app_debug_logger(msg, *args, **kwargs)
            if isinstance(msg, str):
                update_log(msg)

        Logger.debug = debug

    @on(ChatTurn.ChatTurnClicked)
    def click_chat_turn(self, event: events) -> None:
        chatturn = event.widget

        # copy contents of chatbox to clipboard
        pyperclip.copy(chatturn.message)

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
        self.post_message(self.EndChatTurn(role="user"))

    def on_key(self, event: events.Key) -> None:
        """Write Key events to log."""
        pass
        # text_log = self.query_one(TextLog)
        # text_log.write(event)
        # self.post_message(self.AddToChatMessperage("keypress"))

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.dark = not self.dark

    def action_toggle_debug(self) -> None:
        """An action to toggle debug mode."""
        self._show_debug = not self._show_debug

    def action_confirm_y_n(
        self, message: str, confirm_action: str, noconfirm_action: str, title: str = ""
    ) -> None:
        """Build Yes/No Modal Dialog and process callbacks."""
        dialog = self.query_one("#modal_dialog", Dialog)
        dialog.confirm_action = confirm_action
        dialog.noconfirm_action = noconfirm_action
        dialog.set_message(message)
        dialog.show_dialog()

    def action_select_file(
        self,
        message: str = "Select a file to open",
        confirm_action: str = "close_dialog",
        noconfirm_action: str = "close_dialog",
    ) -> None:
        """Open the file picker dialog"""
        file_picker = self.query_one("#file_picker", FilePickerDialog)
        file_picker.message = message
        file_picker.confirm_action = confirm_action
        file_picker.noconfirm_action = noconfirm_action
        file_picker.allowed_extensions = [".pdf"]
        file_picker.show_dialog()

    def action_my_quit(self) -> None:
        """Quit the app."""
        self.log.debug("Quitting")
        print("Quitting")
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

    # def count_tokens(self, chain, query):
    #     with get_openai_callback() as cb:
    #         result = chain.run(query)
    #     print(result)
    #     print(f"Spent a total of {cb.total_tokens} tokens\n")
    #     return result

    def watch__show_debug(self, show_debug: bool) -> None:
        """When __show_debug changes, toggle the class the debug widget."""
        self.app.set_class(show_debug, "-show-debug")

    def set_persona(self, persona: str, model_name: str = "", temperature: float = 0.0):
        """Set the persona and reinitialize the conversation chain."""

        self.current_persona = persona
        if persona not in self.personas:
            raise ValueError(f"Persona '{persona}' not found")

        if model_name == "":
            model_name = self.llm_model_name

        self.conversation = self.ag.new_conversation(
            persona=persona, model_id=model_name, temperature=temperature
        )

    # Add addtional retry logic to the ask_question function
    @retry(tries=3, delay=1)
    async def _ask_question_to_llm(self, question: str, callbacks: List[Callable]):
        await self.conversation.arun(question, callbacks=callbacks)

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
            self.post_message(
                self.AddToChatMessage(
                    role="assistant",
                    message=(
                        "Available Commands:\n"
                        " - new: start a new session\n"
                        " - personas: show available personas\n"
                        " - persona <persona>: set the persona\n"
                        " - models: show available models\n"
                        " - model <model>: set the model\n"
                        " - temperature <temperature>: set the temperature\n"
                        " - summary: summarize the conversation\n"
                        " - dall-e <prompt>: generate an image from the prompt\n"
                    ),
                )
            )
            self.post_message(self.EndChatTurn(role="meta"))
            return

        # if the question is 'new' or 'new session" start a new session
        if question.lower() == "new" or question == "new session":
            # if the session we're in is empty, don't start a new session
            if len(self.record.turns) == 0:
                self.post_message(
                    self.AddToChatMessage(
                        role="assistant",
                        message="You're already in a new session",
                    )
                )
                self.post_message(self.EndChatTurn(role="meta"))
                return

            # clear the chatboxes from the chat container
            self.chat_container.remove_children()

            # start a new history session
            history = self.query_one(HistoryContainer)
            self.record = await history.new_session()
            self.ag.clear_memory()
            self.set_persona(self.default_persona)
            self.llm_model_name = self.mm.default_chat_model
            self.llm_temperature = self.mm.default_chat_temperature
            self._reinitialize_llm_model()
            return

        # if the question is either 'persona', or 'personas' show the available personas
        if question == "personas" or question == "persona":
            self.post_message(
                self.AddToChatMessage(role="assistant", message="Available Personas:\n")
            )
            for persona in self.personas:
                self.post_message(
                    self.AddToChatMessage(role="assistant", message=f" - {persona}\n")
                )
            self.post_message(self.EndChatTurn(role="meta"))
            return

        if question.startswith("persona"):
            # load the new persona
            persona = question.split(maxsplit=1)[1].strip()
            if persona not in self.personas:
                self.post_message(
                    self.AddToChatMessage(
                        role="assistant", message=f"Persona '{persona}' not found"
                    )
                )
                self.post_message(self.EndChatTurn(role="meta"))
                return
            self.post_message(
                self.AddToChatMessage(
                    role="assistant", message=f"Setting persona to '{persona}'"
                )
            )
            self._current_question = ""
            self.set_persona(persona=persona)
            self.post_message(self.EndChatTurn(role="assistant"))
            return

        # if the question is 'models', or 'model', show available models
        if question == "models" or question == "model":
            self.post_message(
                self.AddToChatMessage(role="assistant", message="Available Models:\n")
            )
            self.post_message(
                self.AddToChatMessage(role="assistant", message="- LLM Models:\n")
            )
            for model in self.available_llm_models:
                self.post_message(
                    self.AddToChatMessage(role="assistant", message=f"   - {model}\n")
                )
            self.post_message(
                self.AddToChatMessage(role="assistant", message="- Image Models:\n")
            )
            for model in self.available_image_models:
                self.post_message(
                    self.AddToChatMessage(role="assistant", message=f"   - {model}\n")
                )
            self._current_question = ""
            self.post_message(self.EndChatTurn(role="meta"))
            return

        # if the question starts with 'model', set the model
        if question.startswith("model"):
            # get the model
            model_name = question.split(maxsplit=1)[1].strip()

            # check to see if the name is an llm or image model
            if model_name in self.available_llm_models:
                self.llm_model_name = model_name
                self._reinitialize_llm_model()
                self.post_message(
                    self.AddToChatMessage(
                        role="assistant",
                        message=f"LLM Model set to {self.llm_model_name}",
                    )
                )
                self._current_question = ""
                self.post_message(self.EndChatTurn(role="assistant"))
                return
            elif model_name in self.available_image_models:
                self.image_model_name = model_name
                self._reinitialize_image_model()
                self.post_message(
                    self.AddToChatMessage(
                        role="assistant",
                        message=f"Image model set to {self.image_model_name}",
                    )
                )
                self._current_question = ""
                self.post_message(self.EndChatTurn(role="assistant"))
                return
            else:
                self.post_message(
                    self.AddToChatMessage(
                        role="assistant",
                        message=f"Model '{model_name}' not found",
                    )
                )
                self._current_question = ""
                self.post_message(self.EndChatTurn(role="meta"))
                return

        # if the question starts with 'summary', summarize the conversation
        if question.startswith("summary"):
            # get the summary
            summary = str(self.memory.load_memory_variables({}))
            # post the summary
            self._current_question = ""
            self.post_message(self.AddToChatMessage(role="assistant", message=summary))
            self.post_message(self.EndChatTurn(role="meta"))
            return

        # if the question starts with 'temperature', set the temperature
        if question.startswith("temperature"):
            # get the temperature
            self.llm_temperature = float(question.split(maxsplit=1)[1].strip())
            # post the summary
            self.post_message(
                self.AddToChatMessage(
                    role="assistant",
                    message=f"Temperature set to {self.llm_temperature}",
                )
            )
            self._current_question = question
            self._reinitialize_llm_model()
            self.post_message(self.EndChatTurn(role="assistant"))
            return

        # if question starts with 'dall-e ' pass to Dall-e
        if question.startswith("dall-e "):
            question = question[7:]
            self.post_message(
                self.AddToChatMessage(role="assistant", message="Generating...")
            )
            self.post_message(self.EndChatTurn(role="meta"))
            self._current_question = question
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
            self.post_message(self.AddToChatMessage(role="assistant", message=out))
            self.post_message(self.EndChatTurn(role="assistant"))
            return

        # Just a normal question at this point
        self._current_question = question

        # ask the question and wait for a response, the routine is responsible for
        # posting any tokens to the chatbox via the callback function passed when the
        # object was created
        try:
            await self.ag.ask(question)
        except Exception as e:
            self.post_message(
                self.AddToChatMessage(
                    role="assistant", message=f"Error running autogen: {e}"
                )
            )

        # Done with response; clear the chatbox
        self.scroll_to_end()
        self.post_message(self.EndChatTurn(role="assistant"))

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Called when the worker state changes."""

        # if the worker is done, scroll to the end of the chatbox
        # if event.state == Worker.State.SUCCESS:
        #     self.post_message(self.EndChatTurn(role="user"))
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
        def __init__(self, role: str, message: str) -> None:
            assert role in ["user", "assistant"]
            self.role = role
            self.message = message
            super().__init__()

    @on(AddToChatMessage)
    async def add_to_chat_message(self, chat_token: AddToChatMessage) -> None:
        chunk = chat_token.message
        role = chat_token.role

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
            self.chatbox = ChatTurn(role=role)
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
        def __init__(self, role: str) -> None:
            assert role in ["user", "assistant", "meta"]
            self.role = role
            super().__init__()

    @on(EndChatTurn)
    async def end_chat_turn(self, event: EndChatTurn) -> None:
        """Called when the worker state changes."""

        # some situations can cause EndChatTurn to be successively called, so there
        # may not be an existing chatbox on the second call
        if self.chatbox is None:
            self.log("Received EndChatTurn message with no active chatbox")
            return

        # If we hae a response, add the turn to the conversation record
        if event.role == "assistant":
            self.record.add_turn(
                persona=self.current_persona,
                prompt=self._current_question,
                response=self.chatbox.message,
                # summary=self.memory.moving_summary_buffer,
                summary=self.ag.memory,
                model=self.llm_model_name,
                temperature=self.llm_temperature,
                # memory_messages=messages_to_dict(self.ag.memory),
                memory_messages=self.ag.memory,
            )

            history = self.query_one(HistoryContainer)
            await history.update_conversation(self.record)

        # Update debug pane
        debug_pane = self.query_one(DebugPane)
        debug_pane.update_status()

        # if we have a chatbox, close it.
        self.chatbox = None

    @on(HistoryContainer.HistorySessionClicked)
    def _on_history_session_clicked(
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
        else:
            # load the parameters from the last turn and reinitialize
            self.current_persona = self.record.turns[-1].persona
            self.llm_model_name = self.record.turns[-1].model
            self.llm_temperature = self.record.turns[-1].temperature
            self._current_question = self.record.turns[-1].prompt
            self._reinitialize_llm_model(messages=self.record.turns[-1].memory_messages)

        # clear the chatboxes from the chat container
        self.chat_container.remove_children()

        # load the chat history from the record
        for turn in self.record.turns:
            self.post_message(self.AddToChatMessage(role="user", message=turn.prompt))
            self.post_message(self.EndChatTurn(role="meta"))
            self.post_message(
                self.AddToChatMessage(role="assistant", message=turn.response)
            )
            self.post_message(self.EndChatTurn(role="meta"))


def run():
    """Run the app"""
    app.run()


app = ChatApp()

if __name__ == "__main__":
    run()
