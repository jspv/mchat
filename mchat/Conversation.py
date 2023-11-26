from dataclasses import dataclass, asdict, field
from datetime import datetime
from copy import deepcopy
import json
import uuid


@dataclass
class Turn(object):
    persona: str
    prompt: str
    response: str
    model: str
    summary: str
    temperature: float
    memory_messages: list = field(default_factory=list)
    datetime: datetime = field(default_factory=lambda: datetime.now())

    def to_json(self):
        data = asdict(self)
        # convert datetime to string, including date
        data["datetime"] = data["datetime"].isoformat()
        return json.dumps(data)

    @staticmethod
    def from_json(json_string):
        data = json.loads(json_string)
        # convert datetime string to datetime object
        data["datetime"] = datetime.fromisoformat(data["datetime"])
        return Turn(**data)


# object to store conversation details
@dataclass
class ConversationRecord(object):
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    summary: str = ""
    turns: list = field(default_factory=list[Turn])

    def add_turn(self, **kwargs):
        self.turns.append(Turn(**kwargs))

    def copy(self):
        new_record = deepcopy(self)
        new_record.id = str(uuid.uuid4())
        return new_record

    def to_json(self) -> str:
        out = {}
        out["id"] = self.id
        out["summary"] = self.summary
        out["turns"] = [turn.to_json() for turn in self.turns]
        return json.dumps(out)

    @staticmethod
    def from_json(json_string):
        in_json = json.loads(json_string)
        in_json["turns"] = [Turn.from_json(turn) for turn in in_json["turns"]]
        return ConversationRecord(**in_json)
