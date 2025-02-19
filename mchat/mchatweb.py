import argparse
import asyncio
import json
import os
from typing import AsyncIterator, Literal, Optional, Sequence

import yaml
from nicegui import events, ui

from config import settings
from mchat.history import HistoryContainer, HistorySessionBox
from mchat.llm import AutogenManager, LLMTools, ModelManager

DEFAULT_AGENT_FILE = "mchat/default_agents.yaml"
EXTRA_AGENTS_FILE = settings.get("extra_agents_file", None)


class FakeResponse:
    def __init__(self, text: str, delay: Optional[float] = 0.2):
        self.text = text
        self.delay = delay
        tokens = text.split()
        # prepend spaces from the second token onwards
        self.tokens = iter([tokens[0]] + [f" {word}" for word in tokens[1:]])

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        try:
            await asyncio.sleep(self.delay)  # Simulate asynchronous behavior
            return next(self.tokens)
        except StopIteration:
            raise StopAsyncIteration from None


class ChatTurn(object):
    def __init__(
        self,
        container: ui.element,
        question: str = "",
        role: str = "user",
        title: str = "Agent",
    ):
        self.role = role
        self.agent = title  # The agent name
        self.response: str = ""
        self.container = container
        self._in_progress: bool = False

        with self.container:
            if question and role == "user":
                ui.chat_message(text=question, name="You", sent=True)
            self.response_chat = ui.chat_message(name=self.agent, sent=False)
            # this keeps the label invisible until we get content
            self.response_chat.visible = False
            self.in_progress = True

    @property
    def in_progress(self) -> bool:
        return self._in_progress

    @in_progress.setter
    def in_progress(self, value: bool):
        if self._in_progress == value:
            return
        self._in_progress = value
        if self._in_progress:
            with self.container:
                self.spinner = ui.spinner(type="dots")
        else:
            self.container.remove(self.spinner)

    async def append_chunk(self, chunk: str):
        self.response += chunk
        self.response_chat.clear()
        if self.response_chat.visible is False:
            self.response_chat.visible = True
        with self.response_chat:
            ui.markdown(
                self.response, extras=["fenced-code-blocks", "tables", "code-friendly"]
            )

        # force scroll to the bottom
        ui.run_javascript("window.scrollTo(0, document.body.scrollHeight)")


class WebChatApp:
    # The main area for chat communicaitons
    _message_container: ui.element | None = None

    # stubbing out logging
    log = print

    def run(self):
        ui.run()

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

    async def send(self, question: str) -> AsyncIterator[str]:
        responder = FakeResponse(
            "This is a fake response from the chatbot. This is going to be really long, hopefully long engough to cause a wrap."
        )
        async for chunk in responder:
            yield chunk

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
            and agent_name != self.current_agent
            and role == "assistant"
        ):
            agent_name = f"{self.current_agent}:{agent_name}"

        else:
            agent_name = self.current_agent

        # if the role is 'assistant_historical', change it to 'assistant'
        if role == "assistant_historical":
            role = "assistant"

        # Todo - find out why this is happening
        if message is None:
            return

        if self.chatbox is None:
            if role == "user":
                self.chatbox = ChatTurn(
                    self._message_container,
                    role=role,
                    title=agent_name,
                    question=message,
                )
            else:
                self.chatbox = ChatTurn(
                    self._message_container, role=role, title=agent_name
                )
                await self.chatbox.append_chunk(message)
        else:
            await self.chatbox.append_chunk(message)

    @property
    def in_progress(self) -> bool:
        if self.chatbox is None:
            return False
        return self.chatbox.in_progress

    @in_progress.setter
    def in_progress(self, value: bool):
        if self.chatbox is not None:
            self.chatbox.in_progress = value

    async def end_chat_turn(self, role: str, agent_name: str | None = None):
        """Marks the end of a portion of the conversation and stores it"""
        assert role in ["user", "assistant", "meta"]

        if self.chatbox is None:
            self.log(
                f"Received end_chat_turn with no active chatbox. role={role}, "
                f"agent_name={agent_name}"
            )
            return

        # note if we're seeing a response from a sub-agent
        if agent_name is not None and agent_name != self.current_agent:
            agent_name = f"{self.current_agent}:{agent_name}"
        else:
            agent_name = self.current_agent

        # if the role is 'user', this is a new message so setup a new turn
        if role == "user":
            self.record.new_turn(
                agent=agent_name,
                prompt=self._current_question,
                model=self.llm_model_name,
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
        if self.chatbox is not None:
            self.chatbox.in_progress = False
        self.chatbox = None

    async def ask_question(self, question: str):
        """Read the user's question and take action"""
        if question.lower() == "help":
            await self.end_chat_turn(role="meta")
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
            await self.end_chat_turn(role="user")
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
            self._message_container.clear()
            self.record = await self.history_container.new_session()
            self.llm_model_name = self.mm.default_chat_model
            self.llm_temperature = self.mm.default_chat_temperature
            self.current_agent = self.default_agent
            await self.set_agent_and_model(
                agent=self.default_agent,
                model=self.mm.default_chat_model,
                update_ui=True,
            )
            return

        # agent
        if question.lower() == "agents" or question.lower() == "agent":
            await self.end_chat_turn(role="meta")
            await self.add_to_chat_message(
                role="assistant", message="Available agents\n\n", agent_name="meta"
            )

            for agent in [
                agent_name
                for agent_name, val in self.agents.items()
                if val.get("chooseable", True)
            ]:
                if agent == self.current_agent:
                    await self.add_to_chat_message(
                        role="assistant",
                        message=f" - *{agent}* (current)\n",
                        agent_name="meta",
                    )

                else:
                    (
                        await self.add_to_chat_message(
                            role="assistant", message=f" - {agent}\n", agent_name="meta"
                        ),
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
            await self.set_agent_and_model(
                agent=agent, model_context="preserve", update_ui=True
            )
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

            self._current_question = ""
            await self.end_chat_turn(role="meta")
            return

        # set model
        if question.startswith("model"):
            await self.end_chat_turn(role="meta")
            # get the model
            model = question.split(maxsplit=1)[1].strip()

            # check to see if the name is an llm or image model
            if model in self.available_llm_models:
                self.llm_model_name = model
                # self.log.debug(f"switching to llm model {model}")
                try:
                    await self.set_agent_and_model(
                        model=model, model_context="preserve", update_ui=True
                    )
                except ValueError as e:
                    await self.add_to_chat_message(
                        role="assistant",
                        message=f"Error setting model: {e}",
                        agent_name="meta",
                    )

                else:
                    await self.add_to_chat_message(
                        role="assistant",
                        message=f"Model set to {self.llm_model_name}",
                        agent_name="meta",
                    )
                await self.end_chat_turn(role="meta")
                self._current_question = ""
                return
            elif model in self.available_image_models:
                self.image_model_name = model
                # self.log.debug(f"switching to image model {model}")
                self._reinitialize_image_model()
                await self.add_to_chat_message(
                    role="assistant",
                    message=f"Image model set to {self.image_model_name}",
                    agent_name="meta",
                )
                self._current_question = ""
                await self.end_chat_turn(role="meta")
                return
            else:
                await self.add_to_chat_message(
                    role="assistant",
                    message=f"Model '{model}' not found",
                    agent_name="meta",
                )
                self._current_question = ""
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
        await self.end_chat_turn(role="user")
        self._current_question = question
        try:
            await self.ag.ask(question)
        except Exception as e:
            await self.add_to_chat_message(
                role="assistant",
                message=f"Error running autogen: {e}",
                agent_name="meta",
            )

        # Assistant is done
        self.in_progress = False
        await self.end_chat_turn(role="assistant")

        # TODO - fix the spinner to end when we reach this point.

    def __init__(self) -> None:
        # current debug log
        self.debug_log = ""

        self.chatbox: ChatTurn | None = None

        # parse arguments
        self.parse_args_and_initialize()
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
            message_callback=self.on_llm_new_token,
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

        # Initialze the agent, model and active conversation record
        asyncio.run(self.initialize())

        # Define the main UI layout
        @ui.page("/", dark=False)
        def main_ui():
            # the queries below are used to expand the contend down to the footer (content can then use flex-grow to expand)
            ui.query(".q-page").classes("flex")
            ui.query(".nicegui-content").classes("w-full")

            # message_container = ui.scroll_area().classes(
            #     # "w-full max-w-2xl mx-auto flex-grow items-stretch"
            #     "items-stretch"
            # )
            self._message_container = ui.element("div").classes(
                "w-full max-w-4xl mx-auto flex-grow items-stretch"
            )

            # Header
            with (
                ui.header(elevated=True)
                .style("background-color: #3874c8")
                .classes("items-center justify-between")
            ):
                ui.label("HEADER")
                ui.button(on_click=lambda: right_drawer.toggle(), icon="menu").props(
                    "flat color=white"
                )

            # Left drawer History Container, this will aslo load the database and
            # populate the history container
            self.history_container = HistoryContainer(
                select_callback=self.history_session_clicked, new_label="New Session"
            )
            # Load the active conversation record
            self.record = self.history_container.active_record

            # Right Drawer Debug Pane
            with (
                ui.right_drawer(value=False, fixed=True)
                .style("background-color: #ebf1fa")
                .props("bordered") as right_drawer
            ):
                ui.label("DEBUG")

            # Footer and Input Area
            with ui.footer().style("background-color: #3874c8"):
                with ui.row().classes("w-full no-wrap items-center"):
                    placeholder = "Type your question here"
                    with ui.textarea(placeholder=placeholder) as text:

                        async def _enter(e: events.GenericEventArguments):
                            if e.args["shiftKey"]:
                                return
                            else:
                                # this is a new quesiton
                                question = text.value
                                text.value = ""
                                with self._message_container:
                                    await self.add_to_chat_message(
                                        role="user", message=question
                                    )
                                await self.ask_question(question)
                                # await self.end_chat_turn(role="user")

                        text.props("rounded outlined input-class=mx-3")
                        text.classes("w-full self-center")
                        text.props('input-class="h-14"')
                        # text.on("keydown.enter", send)
                        text.on("keydown.enter", _enter)

    async def history_session_clicked(self, click_args: dict):
        """Callback for when a history session is clicked"""
        box = click_args["box"]
        record = click_args["record"]

        self.record = record

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
        self._message_container.clear()

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
        # self.scroll_to_end()

    async def initialize(self):
        # set the agent and model
        self.default_agent = getattr(settings, "defaults.agent", "default")

        await self.set_agent_and_model(
            agent=self.default_agent,
            model=self.llm_model_name,
            temperature=self.llm_temperature,
        )

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

        # if model_context == "preserve":
        #     model_context = await self.ag.get_memory()

        self.conversation = await self.ag.new_conversation(
            agent=agent, model_id=model, temperature=temperature
        )

        # # if it is a new agent, load compatable models into statusbar
        # if agent != getattr(self, "current_agent", None):
        #     self.query_one(StatusBar).load_models(
        #         [(model, model) for model in compatible_models],
        #         value=model,
        #     )

        # TODO rename llm_model_name to current_model
        self.current_agent = agent
        self.llm_model_name = model
        self.llm_temperature = temperature

        # update the memory with the model context
        # if model_context:
        #     await self.ag.update_memory(model_context)

        self._current_question = ""

        # # if setting streaming is supported, enable the streaming selector
        # if self.ag.stream_tokens is not None:
        #     self.query_one(StatusBar).enable_stream_selector()
        #     self.query_one(StatusBar).set_streaming(self.ag.stream_tokens)
        # else:
        #     self.query_one(StatusBar).disable_stream_selector()

        # # update the agent in the status bar if needed
        # # (e.g. if the agent was set by command)
        # if update_ui and self.query_one(StatusBar).agent != agent:
        #     self.query_one(StatusBar).agent = agent

        # # show the current model if not already set
        # if update_ui and self.query_one(StatusBar).model != model:
        #     self.query_one(StatusBar).model = model

        # self.log.debug(f"Agent set to {agent}")
        # self.log.debug(f"Model set to {self.ag.model}")

        # debug_pane = self.query_one(DebugPane)
        # # debug_pane.update_status()
        # # debug_pane.update_entry(
        # #     "summary_buffer",
        # #     lambda: self.memory.moving_summary_buffer,
        # # )

        # await debug_pane.update_status()


if __name__ in {"__main__", "__mp_main__"}:
    app = WebChatApp()
    app.run()
