from dataclasses import dataclass


# object to store conversation details
class ConversationRecord(object):
    def __init__(self):
        self.turns = []

    def add_turn(self, persona, prompt, response, summary, model, temperature, memory):
        self.turns.append(
            Turn(
                persona=persona,
                prompt=prompt,
                response=response,
                summary=summary,
                model=model,
                temperature=temperature,
                memory=memory,
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
            out += f"memory: {turn.memory}\n"
            out += "\n"
        return out

    def log_last(self):
        turn = self.turns[-1]
        out = ""
        out += f"persona: {turn.persona}\n"
        out += f"prompt: {turn.prompt}\n"
        out += f"response: {turn.response}\n"
        out += f"summary: {turn.summary}\n"
        out += f"model: {turn.model}\n"
        out += f"temperature: {turn.temperature}\n"
        out += f"memory: {turn.memory}\n"
        return out


@dataclass
class Turn(object):
    persona: str
    prompt: str
    response: str
    model: str
    summary: str
    temperature: float
    memory: dict
