import os
import asyncio
import argparse

from langchain.callbacks import get_openai_callback
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler


from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, TextLog, Label
from textual import on
from textual import events
from textual.worker import Worker
from rich.table import Table
from rich.syntax import Syntax
from rich.text import Text
from rich.align import Align

from typing import Any, Dict, List

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage, HumanMessage, AIMessage, LLMResult
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory


# Tracing settings for debugging
# os.environ["LANGCHAIN_TRACING"] = "true"
# os.environ["LANGCHAIN_HANDLER"] = "langchain"
# os.environ["LANGCHAIN_SESSION"] = "callback_testing"  # This is session

EXTRA_PERSONA_FILE = "personas.json"


class MyCustomAsyncHandler(AsyncCallbackHandler):
    """Async callback handler that can be used to handle callbacks from langchain."""

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""
        print("zzzz....")
        await asyncio.sleep(0.3)
        # class_name = serialized["name"]
        print("Hi! I just woke up. Your llm is starting")

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"async: {token}")

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when chain ends running."""
        print("zzzz....")
        await asyncio.sleep(0.3)
        print("Hi! I just woke up. Your llm is ending")


class StreamTokenCallback(AsyncCallbackHandler):
    """Async callback handler that can will push tokens to the.

    ui_update_callback: Callable - a callback that will be called with the token
    """

    def __init__(self, ui_update_callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui_update_callback = ui_update_callback

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.ui_update_callback.write(token)
        self.ui_update_callback.write(
            "blah",
        )

        print(f"Sending update: {token}")


class GetTokenSyncHandler(BaseCallbackHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        raise NotImplementedError("Sync")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        raise NotImplementedError("end - via sync")


class ChatApp(App):
    BINDINGS = [("d", "toggle_dark", "Toggle dark mode")]

    def __init__(self, tui: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # parse arguments
        self.parse_args_and_initialize()

        if "OPENAI_API_KEY" not in os.environ:
            # set the environment variable to the contents of the file
            os.environ["OPENAI_API_KEY"] = open("openai_key.private").read().strip()

        # Initialize the language model
        self.llm_model_name = "gpt-3.5-turbo"
        self.llm_temperature = 0.7
        if self.tui is True:
            self.llm = ChatOpenAI(
                model_name=self.llm_model_name,
                verbose=False,
                streaming=True,
                temperature=self.llm_temperature,
            )
        else:
            self.llm = ChatOpenAI(
                model_name=self.llm_model_name,
                verbose=False,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()],
                temperature=self.llm_temperature,
            )

        # Initialize the summary model
        self.summary_model_name = "gpt-3.5-turbo"
        self.summary_temperature = 0.1
        self.summary_llm = ChatOpenAI(
            model_name=self.summary_model_name, temperature=self.summary_temperature
        )

        self.personas = {
            "financial manager": {
                "description": (
                    "I am a skilled financial manager familiar with analysis and"
                    " investing in equities, fixed income, alternative investments, and"
                    " real estate. I am familiar with the use of derivatives and"
                    " options to hedge risk and enhance returns. I am familiar with the"
                    " use of leverage to enhance returns. I am familiar with the use of"
                    " leverage to hedge risk. I am also an excellent mentor and when I"
                    " use financial jargon, I will always provide a clear definition"
                    " for the jargon terms at the end of my response"
                ),
                "extra_context": [],
            },
            "default": {
                "description": (
                    "I am a highly intelligent question answering bot. If you ask me a"
                    " question that is rooted in truth, I will give you the answer. If"
                    " you ask me a question that is nonsense, trickery, or has no clear"
                    " answer, I will respond with a nonsenseresponse."
                ),
                "extra_context": [],
            },
            "linux computer": {
                "description": (
                    "Act as a Linux terminal. I will type commands and you will reply"
                    " with what the terminal should show. Only reply with the terminal"
                    " output inside one unique code block, and nothing else. Do not"
                    " write explanations. Do not type commands unless I instruct you to"
                    " do so. When I need to tell you something that is not a command I"
                    " will do so by putting text inside square brackets [like this]."
                ),
                "extra_context": [
                    [
                        "human",
                        "hostname",
                    ],
                    ["ai", "```linux-terminal```"],
                ],
            },
            "blank": {
                "description": "",
                "extra_context": [],
            },
        }

        # if there is an EXTRA_PERSONA_FILE, load the personas from there
        if os.path.exists(EXTRA_PERSONA_FILE):
            import json

            with open(EXTRA_PERSONA_FILE) as f:
                extra_personas = json.load(f)
            self.personas.update(extra_personas)

        self.memory = ConversationSummaryBufferMemory(
            llm=self.summary_llm, max_token_limit=1000, return_messages=True
        )

        self.set_persona("default")

    def parse_args_and_initialize(self):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "-t",
            "--text",
            help="use straight text instead of the TUI. ",
            action="store_true",
        )

        parser.add_argument(
            "-v", "--verbose", help="Increase verbosity", action="store_true"
        )

        args, unknown = parser.parse_known_args()

        if args.text:
            self.tui = False
        else:
            self.tui = True

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield TextLog(highlight=True, markup=True)
        yield Label()
        yield Input()
        yield Footer()

    def on_ready(self) -> None:
        """Called  when the DOM is ready."""
        text_log = self.query_one(TextLog)
        label = self.query_one(Label)
        input = self.query_one(Input)

        label.update("Press [b]Enter[/] to start chatting!")
        text_log.write(Syntax("This is stuff", "python", indent_guides=True))
        text_log.write("[bold magenta]Write text or any Rich renderable!")
        input.focus()

    @on(Input.Submitted)
    def submit_question(self, event: events) -> None:
        text_log = self.query_one(TextLog)
        input = self.query_one(Input)
        text_log.write(f"[bold blue]{event.value}")
        input.value = ""
        self.run_worker(self.ask_question(event.value), exit_on_error=True)
        # self.run_worker(
        #     self.conversation.arun(event.value, callbacks=[GetTokenSyncHandler()]),
        #     exit_on_error=True,
        #     exclusive=True,
        # )

    def on_key(self, event: events.Key) -> None:
        """Write Key events to log."""
        text_log = self.query_one(TextLog)
        text_log.write(event)

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.dark = not self.dark

    def count_tokens(self, chain, query):
        with get_openai_callback() as cb:
            result = chain.run(query)
        print(result)
        print(f"Spent a total of {cb.total_tokens} tokens\n")

        return result

    # there is a bug in actually using templates with the memory object, so we
    # build the prompt template manually

    # build the prompt template; note: the MessagesPlaceholder is required
    # to be able to access the history of messages, its variable "history" will be
    # replaced with the history of messages by the conversation chain as provided by
    # the memory object.
    def build_prompt_template(self, persona):
        base = []

        # Initial system message
        if len(self.personas[persona]["description"]) > 0:
            base.append(
                SystemMessagePromptTemplate.from_template(
                    self.personas[persona]["description"]
                )
            )
        # Extra system messages
        for extra in self.personas[persona]["extra_context"]:
            if extra[0] == "ai":
                base.append(AIMessagePromptTemplate.from_template(extra[1]))
            elif extra[0] == "human":
                base.append(HumanMessagePromptTemplate.from_template(extra[1]))
            elif extra[0] == "system":
                base.append(SystemMessagePromptTemplate.from_template(extra[1]))
            else:
                raise ValueError(f"Unknown extra context type {extra[0]}")

        # History Placeholder
        base.append(MessagesPlaceholder(variable_name="history"))

        # Human message
        base.append(HumanMessagePromptTemplate.from_template("{input}"))

        return ChatPromptTemplate.from_messages(base)

    def set_persona(self, persona: str):
        if persona not in self.personas:
            raise ValueError(f"Persona '{persona}' not found")
        # have to rebuild prompt and chain due to
        # https://github.com/hwchase17/langchain/issues/1800 - can't use templates
        prompt = self.build_prompt_template(persona=persona)
        self.conversation = ConversationChain(
            llm=self.llm,
            verbose=False,
            prompt=prompt,
            memory=self.memory,
        )
        self.memory.clear()

    async def ask_question(self, question: str):
        """Ask a question to the AI and return the response."""
        text_log = self.query_one(TextLog)
        await self.conversation.arun(
            question, callbacks=[StreamTokenCallback(text_log)]
        )

    def input_loop(self):
        mutli_line_input = False
        while True:
            if mutli_line_input:
                print(
                    "[Multi-line] Type 'quit' to exit, 'new' to start a new"
                    " conversation, 'ml' to switch off multi-line input.  Press ctrl+d"
                    " or enter '.' on a blank line: to process input."
                )
                query = ""
                while True:
                    # read lines
                    try:
                        line = input()
                    except EOFError:
                        break
                    if line == ".":
                        break
                    query += line + "\n"
                query = query.strip()
            else:
                query = input(
                    "Type 'quit' to exit, 'new' to start a new conversation, 'ml' to"
                    " switch to multi-line input: "
                )

            if query == "quit":
                exit()
            if query == "new":
                print("Starting a new conversation...")
                self.memory.clear()
            elif query == "ml":
                mutli_line_input = not mutli_line_input

            # if query starts with 'persona', or 'personas' get the persona, if it no persona
            # is specified, show the available personas
            elif query == "personas" or query == "persona":
                print("Available personas:")
                for persona in self.personas:
                    print(f" - {persona}")
            elif query.startswith("persona"):
                # load the new persona
                persona = query.split(maxsplit=1)[1].strip()
                if persona not in self.personas:
                    print(f"Persona '{persona}' not found")
                    continue
                print(f"Setting persona to '{persona}'")
                self.set_persona(persona=persona)

            else:
                self.conversation.run(query)
                print("\n")

    def run(self, *args, **kwargs):
        """Run the app."""

        if self.tui:
            super().run(*args, **kwargs)
        else:
            self.input_loop()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Called when the worker state changes."""
        self.log(event)


if __name__ == "__main__":
    app = ChatApp()
    app.run()
