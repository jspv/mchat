from __future__ import annotations

import json
import logging
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from . import logging_config as _enable_trace  # noqa: F401

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    """Object to store a single back/forth turn in a conversation."""

    agent: str
    prompt: str
    model: str
    summary: str = ""
    temperature: float = 0.0
    responses: list[dict[str, Any]] = field(default_factory=list)
    memory_messages: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_json(self) -> str:
        """Convert Turn object to a JSON string."""
        data = asdict(self)
        # Convert timestamp to ISO format string
        data["timestamp"] = data["timestamp"].isoformat()
        return json.dumps(data)

    @staticmethod
    def from_json(json_string: str) -> Turn:
        """Create a Turn object from a JSON string."""
        data = json.loads(json_string)
        # Convert timestamp string back to datetime object
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return Turn(**data)


@dataclass
class ConversationRecord:
    """Object to store a conversation."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created: datetime = field(default_factory=datetime.now)
    summary: str = ""
    turns: list[Turn] = field(default_factory=list)

    def new_turn(self, **kwargs) -> None:
        """Create a new turn in the conversation."""
        self.turns.append(Turn(**kwargs))

    def add_to_turn(
        self,
        agent_name: str,
        response: str,
        memory_messages: list[Any] | None = None,
    ) -> None:
        """Add a response to the most recent turn in the conversation."""
        logger.trace(
            f"Adding response to turn. Agent: {agent_name}, Response: {response}, "
            f"memory_messages: {memory_messages}"
        )
        if not self.turns:
            logger.error(
                f"No existing turns to add a response to. Agent: "
                f"{agent_name}, Response: {response}"
            )
            raise IndexError("No existing turns to add a response to.")
        self.turns[-1].responses.append({"agent": agent_name, "response": response})
        if memory_messages is not None:
            self.turns[-1].memory_messages = memory_messages
        logger.trace("end of add_to_turn")

    def copy(self) -> ConversationRecord:
        """Create a deep copy of the conversation record with a new ID and timestamp."""
        new_record = deepcopy(self)
        new_record.id = str(uuid.uuid4())
        new_record.created = datetime.now()
        return new_record

    def to_json(self) -> str:
        """Convert ConversationRecord object to a JSON string."""
        out = {
            "id": self.id,
            "summary": self.summary,
            "created": self.created.isoformat(),
            "turns": [turn.to_json() for turn in self.turns],
        }
        return json.dumps(out)

    @staticmethod
    def from_json(json_string: str) -> ConversationRecord:
        """Create a ConversationRecord object from a JSON string."""
        data = json.loads(json_string)
        data["created"] = datetime.fromisoformat(data["created"])
        data["turns"] = [Turn.from_json(turn_json) for turn_json in data["turns"]]
        return ConversationRecord(
            id=data.get("id", str(uuid.uuid4())),
            created=data["created"],
            summary=data.get("summary", ""),
            turns=data["turns"],
        )
