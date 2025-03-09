import logging
from datetime import datetime, timedelta, timezone
from typing import Callable, TypeAlias

import apsw.bestpractice
from nicegui import ui

from mchat.conversation import ConversationRecord
from mchat.llm import LLMTools
from mchat.styles import colors as c

logger = logging.getLogger(__name__)


# Define type alias for Callbacks
CallbackType: TypeAlias = Callable[..., None] | None


class HistorySessionBox(object):
    """A container for a single chat history session"""

    _callback: CallbackType = None  # Shared among all instances

    def __init__(
        self,
        record: ConversationRecord,
        new_label: str = "",
        label: str = "",
    ) -> None:
        self.new_label = new_label
        self.label = label
        self.record = record
        self._active = False

        session_box_label = (
            self.get_relative_date(self.record.created) if self.record.turns else "..."
        )

        with ui.card().classes(
            f"w-full bg-{c.historycard_l} dark:bg-{c.historycard_d} p-2"
        ) as self.box:
            with ui.row(align_items="center").classes("w-full justify-end"):
                self.boxlabel = ui.label(session_box_label).classes(
                    "justify-self-start"
                )
                ui.element().classes("flex-grow")

                self.copy_button = (
                    ui.button(icon="content_copy")
                    .classes(f"bg-{c.secondary} w-8")
                    .on(
                        "click.stop",
                        lambda: self._call_callback(
                            {"session_box": self, "action": "copy"}
                        ),
                    )
                )
                self.delete_button = (
                    ui.button(icon="delete")
                    .props("color=red")
                    .classes(f"bg-{c.secondary} w-8")
                    .on(
                        "click.stop",
                        lambda: self._call_callback(
                            {"session_box": self, "action": "delete"}
                        ),
                    )
                )

            self.summary = ui.label(
                self.label
                or (
                    self.record.turns[-1].prompt
                    if self.record.turns
                    else self.new_label
                )
            )
            self.copy_button.visible = bool(self.record.turns)

        self.box.on(
            "click",
            lambda: self._call_callback({"session_box": self, "action": "load"}),
        )

    def delete(self) -> None:
        """Remove the HistorySessionBox from the UI"""
        if not self.box.is_deleted:
            self.box.delete()

    @property
    def callback(self) -> CallbackType:
        return type(self)._callback

    @classmethod
    def set_callback(cls, callback: CallbackType) -> None:
        """set the clilck callback for all HistorySessionBox"""
        cls._callback = callback

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, value: bool) -> None:
        self._active = value
        if value:
            self.box.classes(f"!bg-{c.secondary}")
        else:
            self.box.classes(remove=f"!bg-{c.secondary}")

    async def _call_callback(self, *args, **kwargs) -> None:
        """Call the callback if it exists."""
        if self._callback:
            await self._callback(*args, **kwargs)
        else:
            raise ValueError("No callback provided")

    def update_box(self, record: ConversationRecord) -> None:
        """Update the HistorySessionBox with a new ConversationRecord."""
        self.record = record
        self.copy_button.visible = bool(record.turns)

        if record.summary:
            self.summary.text = record.summary
        elif record.turns:
            self.summary.text = record.turns[-1].prompt
            self.record.summary = record.turns[-1].prompt

        self.boxlabel.text = self.get_relative_date(record.created)

    @staticmethod
    def get_relative_date(timestamp: datetime) -> str:
        """Return a relative date label."""
        local_tz = datetime.now(timezone.utc).astimezone().tzinfo
        current_time = datetime.now(local_tz)
        timestamp = timestamp.replace(tzinfo=local_tz)

        if timestamp > current_time:
            return f"future-{timestamp:%m-%d %H:%M}"
        elif timestamp.date() == current_time.date():
            return f"today-{timestamp:%H:%M}"
        elif timestamp.date() == current_time.date() - timedelta(days=1):
            return f"yesterday-{timestamp:%H:%M}"
        elif (current_time - timestamp).days < 7:
            return f"{timestamp:%A}-{timestamp:%H:%M}"
        return f"{timestamp:%m-%d}-{timestamp:%H:%M}"


class HistoryContainer:
    """A container for managing chat history sessions with associated logic."""

    prompt_template = (
        "Here is a list of chat submissions in the form of 'user: message'. "
        "Give me a concise label (≤10 words, max 66 characters) for the conversation. "
        "Chat submissions: {conversation}"
    )

    def __init__(
        self, new_record_callback: CallbackType, new_label: str, app: object
    ) -> None:
        self.sessions: list[HistorySessionBox] = []
        self.new_label = new_label
        self.new_record_callback = new_record_callback
        self.app = app
        self._active_session: HistorySessionBox | None = None

        HistorySessionBox.set_callback(self.history_card_clicked_callback)
        self.connection = self._initialize_db()

        with ui.left_drawer(
            top_corner=True, bottom_corner=True
        ) as self.history_container:
            ui.label("History").style("font-size: 1.5em; font-weight: bold")

            records = [
                self._read_conversation_from_db(row[0])
                for row in self.connection.cursor()
                .execute("SELECT id FROM Conversations")
                .fetchall()
            ]

            # add previous sessions
            for record in sorted(records, key=lambda x: x.created):
                self._add_previous_session(record)

            # add a new session and make it active
            self.active_session = HistorySessionBox(
                record=ConversationRecord(), new_label=self.new_label
            )
            self.sessions.append(self.active_session)

    @property
    def active_record(self) -> ConversationRecord:
        return self._active_session.record

    @property
    def active_session(self) -> HistorySessionBox:
        return self._active_session

    @active_session.setter
    def active_session(self, value: HistorySessionBox) -> None:
        self._active_session = value
        self._active_session.active = True

        for session in self.sessions:
            if session != self._active_session:
                session.active = False

    def _initialize_db(self) -> apsw.Connection:
        """Initialize the SQLite database."""
        apsw.bestpractice.apply(apsw.bestpractice.recommended)
        conn = apsw.Connection("db.db")
        conn.cursor().execute(
            "CREATE TABLE IF NOT EXISTS Conversations (id TEXT PRIMARY KEY, data TEXT)"
        )
        return conn

    def _write_conversation_to_db(self, record: ConversationRecord) -> None:
        """Write or update a conversation record in the database."""
        serialized_conversation = record.to_json()
        self.connection.cursor().execute(
            "INSERT OR REPLACE INTO Conversations (id, data) VALUES (?, ?)",
            (record.id, serialized_conversation),
        )

    def _read_conversation_from_db(self, id: str) -> ConversationRecord:
        row = (
            self.connection.cursor()
            .execute("SELECT data FROM Conversations WHERE id=?", (id,))
            .fetchone()
        )
        return ConversationRecord.from_json(row[0]) if row else ConversationRecord()

    def _delete_conversation_from_db(self, record: ConversationRecord) -> None:
        """Delete a conversation record from the database."""
        self.connection.cursor().execute(
            "DELETE FROM Conversations WHERE id=?", (record.id,)
        )

    def _add_previous_session(self, record: ConversationRecord) -> HistorySessionBox:
        """returns a new HistorySessionBox to the HistoryContainer."""
        with self.history_container:
            self.active_session = HistorySessionBox(
                record=record,
                new_label=self.new_label,
                label=record.summary,
            )
        self.sessions.append(self.active_session)
        # self.active_session.active = True
        return self.active_session

    def _add_session(
        self, record: ConversationRecord | None = None, label: str = ""
    ) -> HistorySessionBox:
        """Add a new session to the HistoryContainer and set it as active."""

        record = record or ConversationRecord()  # Ensure a valid record is used

        with self.history_container:
            self.active_session = HistorySessionBox(
                record=record, new_label=self.new_label, label=label
            )

        self.sessions.append(self.active_session)
        return self.active_session

    async def delete_history_box(self, session_box: HistorySessionBox) -> None:
        """Delete a HistorySessionBox from the HistoryContainer"""
        # check to see if this is the last session
        if self.session_count == 1:
            session_box.delete()
            self._delete_conversation_from_db(session_box.record)
            self.sessions.remove(session_box)
            await self.new_session()
        elif session_box == self.active_session:
            # need to pick a new active session
            pos = self.sessions.index(session_box)
            if pos == 0:
                self.active_session = self.sessions[1]
            else:
                self.active_session = self.sessions[pos - 1]

            session_box.delete()
            self._delete_conversation_from_db(session_box.record)
            self.sessions.remove(session_box)

            # call the callback to load the new active session
            await self.active_session.callback(
                {
                    "action": "load",
                    "session_box": self.active_session,
                    "record": self.active_session.record,
                }
            )
        else:
            session_box.delete()
            self._delete_conversation_from_db(session_box.record)
            self.sessions.remove(session_box)

    async def update_conversation(self, record: ConversationRecord):
        """Update the active session with a new ConversationRecord."""

        # get the last 10 turns
        conversation = [
            {"user": turn.prompt, "ai": turn.responses} for turn in record.turns[-10:]
        ]
        record.summary = await LLMTools.aget_summary_label(conversation)

        # update the active session
        self.active_session.update_box(record)

        # update the database
        self._write_conversation_to_db(record)

    async def new_session(self) -> ConversationRecord:
        """Add a new empty session to the HistoryContainer and set it as active"""
        self._add_session()
        # self.scroll_to_end()
        return self.active_session.record

    async def history_card_clicked_callback(self, click_args: dict) -> None:
        """Callback for when a HistorySessionBox is clicked"""
        action: str = click_args["action"]
        session_box: HistorySessionBox = click_args["session_box"]

        # if the ui is busy, ignore the click
        if self.app.ui_is_busy:
            ui.notify("busy", color="red")
            return

        assert action in ["load", "copy", "delete"]

        if action == "load":
            if self.active_session == session_box:
                return
            self.active_session = session_box
            await self.new_record_callback(session_box.record)
            return

        if action == "delete":
            await self.delete_history_box(session_box)
            await self.new_record_callback(self.active_session.record)
            return

        if action == "copy":
            new_record = session_box.record.copy()
            self._add_session(new_record, label=session_box.label)
            await self.new_record_callback(self.active_session.record)
            return

    @property
    def session_count(self) -> int:
        # return the number of sessions in the history container
        # subtract 1 to account for the header
        return len(self.sessions)
