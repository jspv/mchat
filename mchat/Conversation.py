from __future__ import annotations

from dataclasses import dataclass, asdict, field, InitVar
from typing import ClassVar
from datetime import datetime
from copy import deepcopy
import json
import uuid


@dataclass
class Turn(object):
    """Object to store a single back/forth turn in a conversation"""

    agent: str
    prompt: str
    response: str
    model: str
    summary: str
    temperature: float
    memory_messages: list = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now())

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

    def add_turn(self, **kwargs):
        self.turns.append(Turn(**kwargs))

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
