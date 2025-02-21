from datetime import datetime, timedelta, timezone
from typing import Callable

import apsw.bestpractice
from nicegui import ui

from mchat.Conversation import ConversationRecord
from mchat.llm import LLMTools


class HistorySessionBox(object):
    """A container for a single chat history session"""

    # Class variables since all SessionBox instances will use the same callbacks
    _callback: Callable | None = None

    def __init__(
        self,
        record: ConversationRecord,
        callback: Callable | None = None,
        new_label: str = "",
        label: str = "",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.new_label = new_label
        self.label = label
        self.record = record
        self._active = False

        # callbacks
        self._callback = callback

        if len(self.record.turns) > 0:
            session_box_label = self.get_relative_date(self.record.created)
        else:
            session_box_label = "..."

        with ui.card().classes("w-full bg-secondary") as self.box:
            with ui.row(align_items="center").classes("w-full justify-end"):
                self.boxlabel = ui.label(session_box_label).classes(
                    "justify-self-start"
                )
                ui.element().classes("flex-grow")
                self.copy_button = (
                    ui.button(icon="content_copy")
                    .classes("bg-secondary w-8")
                    .on(
                        "click.stop",
                        lambda: self._call_callback(
                            {"record": self.record, "box": self, "action": "copy"}
                        ),
                    )
                )
                self.delete_button = (
                    ui.button(icon="delete")
                    .props("color=red")
                    .classes("bg-secondary w-8")
                    .on(
                        "click.stop",
                        lambda: self._call_callback(
                            {"record": self.record, "box": self, "action": "delete"}
                        ),
                    )
                )
            self.summary = ui.label()
            if len(self.record.turns) == 0:
                self.summary.text = self.new_label
                self.copy_button.visible = False
            else:
                if len(self.label) > 0:
                    self.summary.text = self.label
                else:
                    self.summary.text = self.record.turns[-1].prompt

        # store a reference to this object in the box
        self.box.history_box = self
        self.box.on(
            "click",
            lambda: self._call_callback(
                {"record": self.record, "box": self, "action": "load"}
            ),
        )

    def delete(self) -> None:
        """Remove the HistorySessionBox from the UI"""
        if self.box.is_deleted is False:
            self.box.delete()

    # Callbacks
    @property
    def callback(self) -> Callable:
        return self._callback

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, value: bool) -> None:
        if value:
            # make all other boxes inactive
            for child in self.box.parent_slot.parent.default_slot.children:
                if hasattr(child, "history_box") and child != self.box:
                    child.history_box.active = False
            self._active = True
            self.box.classes(remove="bg-primary")
            self.box.classes("bg-secondary")
            self.box.update()

        else:
            self._active = False
            self.box.classes(remove="bg-secondary")
            self.box.classes("bg-primary")
            self.box.update()

    @callback.setter
    def callback(self, value: Callable) -> None:
        self._callback = value

    async def _call_callback(self, *args, **kwargs) -> None:
        if self.callback is not None:
            await self.callback(*args, **kwargs)
        else:
            raise ValueError("No callback provided")

    def update_box(self, record: ConversationRecord) -> None:
        """Update the HistorySessionBox with a new ConversationRecord."""

        self.record = record
        self.copy_button.visible = True

        if len(record.summary) > 0:
            self.summary.text = record.summary
            self.boxlabel.text = self.get_relative_date(record.created)
            return

        if len(record.turns) == 0:
            return

        # no summary was providied, so update the summary with the last prompt
        self.summary.text = record.turns[-1].prompt
        self.record.summary = record.turns[-1].prompt

    @staticmethod
    def get_relative_date(timestamp):
        local_timezone = datetime.now(timezone.utc).astimezone().tzinfo
        current_time = datetime.now(local_timezone)
        timestamp = timestamp.replace(tzinfo=local_timezone)
        if timestamp.date() == current_time.date():
            return "today" + "-" + timestamp.strftime("%H:%M")
        elif timestamp.date() == current_time.date() - timedelta(1):
            return "yesterday" + "-" + timestamp.strftime("%H:%M")
        elif (current_time - timestamp).days < 7:
            return timestamp.strftime("%A") + "-" + timestamp.strftime("%H:%M")
        else:
            return timestamp.strftime("%m-%d") + "-" + timestamp.strftime("%H:%M")


class HistoryContainer:
    """A container for managing chat history sessions with asscociated logic"""

    prompt_template = """
    Here is list of chat submissions in the form of 'user: message'. Give me no more
    than 10 words which will be used to remind the user of the conversation.  Your reply
    should be no more than 10 words and at most 66 total characters.
    Chat submissions: {conversation}
    """

    def __init__(
        self,
        callback: Callable | None = None,
        new_label: str = "",
    ) -> None:
        self.new_label = new_label
        HistorySessionBox.callback = callback
        self.connection = self._initialize_db()

        # with ui.left_drawer(top_corner=True, bottom_corner=True).style(
        #     "background-color: #d7e3f4"
        # ) as self.history_container:
        with ui.left_drawer(
            top_corner=True, bottom_corner=True
        ) as self.history_container:
            ui.label("History").style("font-size: 1.5em; font-weight: bold")

            # if there are records in the databaase, load them
            cursor = self.connection.cursor()
            rows = cursor.execute("SELECT id FROM Conversations").fetchall()
            records = []
            if len(rows) > 0:
                for row in rows:
                    records.append(
                        self._read_conversation_from_db(self.connection, row[0])
                    )

                # sort the records by timestamp of first turn
                records.sort(key=lambda x: x.created)
                for record in records:
                    self._add_previous_session(record)

            # create a new session with an empty conversation record
            record = ConversationRecord()
            self.active_session = HistorySessionBox(
                record=record,
                new_label=self.new_label,
            )
            self.active_session.active = True
            # self.active_session.add_class("-active")
            # yield self.active_session

    @property
    def active_record(self) -> ConversationRecord:
        return self.active_session.record

    # Initialize the database and return a connection object
    def _initialize_db(self) -> apsw.Connection:
        # SQLite3 stuff
        apsw.bestpractice.apply(apsw.bestpractice.recommended)

        # Default will create the database if it doesn't exist
        connection = apsw.Connection("db.db")
        cursor = connection.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS Conversations (id TEXT PRIMARY KEY, data TEXT)"
        )
        return connection

    def _write_conversation_to_db(self, conn, conversation):
        cursor = conn.cursor()
        serialized_conversation = conversation.to_json()
        cursor.execute(
            "INSERT OR REPLACE INTO Conversations (id, data) VALUES (?, ?)",
            (
                conversation.id,
                serialized_conversation,
            ),
        )

    def _read_conversation_from_db(self, conn, id) -> ConversationRecord:
        cursor = conn.cursor()
        row = cursor.execute(
            "SELECT data FROM Conversations WHERE id=?", (id,)
        ).fetchone()
        conversation = ConversationRecord.from_json(row[0])
        return conversation

    def _delete_conversation_from_db(self, conn, record: ConversationRecord):
        cursor = conn.cursor()
        id = record.id
        cursor.execute("DELETE FROM Conversations WHERE id=?", (id,))

    def _add_previous_session(self, record: ConversationRecord) -> HistorySessionBox:
        """returns a new HistorySessionBox to the HistoryContainer."""
        with self.history_container:
            self.active_session = HistorySessionBox(
                record=record,
                new_label=self.new_label,
                label=record.summary,
            )
        self.active_session.active = True
        return self.active_session

    async def _add_session(
        self, record: ConversationRecord | None = None, label: str = ""
    ) -> HistorySessionBox:
        """Add a new session to the HistoryContainer and set it as active"""
        label = label if len(label) > 0 else ""

        # if no record is provided, create a new record
        if record is None:
            record = ConversationRecord()

        # Attach the record to a new session
        with self.history_container:
            self.active_session = HistorySessionBox(
                record=record,
                new_label=self.new_label,
                label=label,
            )
        self.active_session.active = True

        # # Activate the new session and deactivate all others
        # self.active_session.add_class("-active")
        # # remove the .-active class from all other boxes
        # for child in self.children:
        #     child.remove_class("-active")
        # await self.mount(self.active_session)
        # pass

    async def delete_history_box(self, session_box: HistorySessionBox) -> None:
        """Delete a HistorySessionBox from the HistoryContainer"""
        # check to see if this is the last session
        if self.session_count == 1:
            session_box.delete()
            await self.delete_conversation(session_box.record)
        elif session_box.record == self.active_record:
            # get the position of the session_box
            pos = self.history_container.default_slot.children.index(session_box.box)
            # if the position is 1, set the next session as active, otherwise set
            # it to the previous session
            if pos == 1:
                self.active_session = self.history_container.default_slot.children[
                    2
                ].history_box
                self.active_session.active = True
            else:
                self.active_session = self.history_container.default_slot.children[
                    pos - 1
                ].history_box
                self.active_session.active = True
            session_box.delete()
            await self.delete_conversation(session_box.record)

            # click the active session to load it
            await self.active_session.callback(
                {
                    "action": "load",
                    "box": self.active_session,
                    "record": self.active_session.record,
                }
            )

        else:
            session_box.delete()
            await self.delete_conversation(session_box.record)

    async def delete_conversation(self, record: ConversationRecord):
        """Delete a conversation from the HistoryContainer and the database"""
        self._delete_conversation_from_db(self.connection, record)
        if self.session_count == 0:
            await self.new_session()

    async def update_conversation(self, record: ConversationRecord):
        """Update the active session with a new ConversationRecord."""

        # grab the last 10 turns
        turns = record.turns
        if len(turns) > 10:
            turns = turns[-10:]

        # group the turns of the conversation and summarize
        conversation = []
        for turn in turns:
            conversation.append({"user": turn.prompt, "ai": turn.responses})
        record.summary = await LLMTools.aget_summary_label(conversation)

        # update the active session
        self.active_session.update_box(record)

        # update the database
        self._write_conversation_to_db(self.connection, record)

    async def new_session(self) -> ConversationRecord:
        """Add a new empty session to the HistoryContainer and set it as active"""
        await self._add_session()
        # self.scroll_to_end()
        return self.active_record

    @property
    def session_count(self) -> int:
        # return the number of sessions in the history container
        # subtract 1 to account for the header
        return len(self.history_container.default_slot.children) - 1
