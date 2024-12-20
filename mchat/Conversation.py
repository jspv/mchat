from __future__ import annotations

import json
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime


@dataclass
class Turn(object):
    """Object to store a single back/forth turn in a conversation"""

    agent: str
    prompt: str
    model: str
    summary: str = ""
    temperature: float = 0
    responses: list = field(default_factory=list)
    memory_messages: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now())
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_json(self):
        """Convert Turn object to JSON string"""
        data = asdict(self)
        # convert timestamp to string, including date
        data["timestamp"] = data["timestamp"].isoformat()
        return json.dumps(data)

    @staticmethod
    def from_json(json_string):
        """Convert JSON string to Turn object"""
        data = json.loads(json_string)
        # convert timestamp string to datetime object
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return Turn(**data)


# object to store conversation details
@dataclass
class ConversationRecord(object):
    """Object to store a conversation"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created: datetime = field(default_factory=lambda: datetime.now())
    summary: str = ""
    turns: list = field(default_factory=list[Turn])

    def new_turn(self, **kwargs) -> None:
        """Create a new turn in the conversation"""
        self.turns.append(Turn(**kwargs))

    def add_to_turn(
        self, agent_name: str, response: str, memory_messages: list = None
    ) -> None:
        self.turns[-1].responses.append({"agent": agent_name, "response": response})
        if memory_messages:
            self.turns[-1].memory_messages = memory_messages

    def copy(self):
        """create a deep copy of the conversation record with a new id and timestamp"""
        new_record = deepcopy(self)
        new_record.id = str(uuid.uuid4())
        new_record.created = datetime.now()
        return new_record

    def to_json(self) -> str:
        """Convert ConversationRecord object to JSON string"""
        out = {}
        out["id"] = self.id
        out["summary"] = self.summary
        out["turns"] = [turn.to_json() for turn in self.turns]
        out["created"] = self.created.isoformat()
        return json.dumps(out)

    @staticmethod
    def from_json(json_string):
        """Convert JSON string to ConversationRecord object"""
        in_json = json.loads(json_string)
        in_json["turns"] = [Turn.from_json(turn) for turn in in_json["turns"]]
        in_json["created"] = datetime.fromisoformat(in_json["created"])
        return ConversationRecord(**in_json)
