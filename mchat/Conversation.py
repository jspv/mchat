from dataclasses import dataclass
from datetime import datetime


# object to store conversation details
class ConversationRecord(object):
    def __init__(self):
        self.turns = []

    def add_turn(
        self, persona, prompt, response, summary, model, temperature, memory_messages
    ):
        self.turns.append(
            Turn(
                persona=persona,
                prompt=prompt,
                response=response,
                summary=summary,
                model=model,
                temperature=temperature,
                memory_messages=memory_messages,
                datetime=datetime.now().time(),
            )
        )

    def log_all(self):
        out = ""
        for turn in self.turns:
            out += f"persona: {turn.persona}\n"
            out += f"prompt: {turn.prompt}\n"
            out += f"response: {turn.response}\n"
            out += f"summary: {turn.summary}\n"
            out += f"model: {turn.model}\n"
            out += f"temperature: {turn.temperature}\n"
            out += f"memory_messages: {turn.memory_messages}\n"
            out += "\n"
        return out

    def log_last(self):
        if len(self.turns) == 0:
            return "No chat turns logged yet."
        turn = self.turns[-1]
        out = ""
        out += f"persona: {turn.persona}\n"
        out += f"prompt: {turn.prompt}\n"
        out += f"response: {turn.response}\n"
        out += f"summary: {turn.summary}\n"
        out += f"model: {turn.model}\n"
        out += f"temperature: {turn.temperature}\n"
        out += f"memory_messages: {turn.memory_messages}\n"
        return out


@dataclass
class Turn(object):
    persona: str
    prompt: str
    response: str
    model: str
    summary: str
    temperature: float
    memory_messages: list
    datetime: datetime.time
