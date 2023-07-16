import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain.callbacks import get_openai_callback

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


os.environ["LANGCHAIN_TRACING"] = "true"
os.environ["LANGCHAIN_HANDLER"] = "langchain"
os.environ["LANGCHAIN_SESSION"] = "agent_chain"  # This is session

EXTRA_PERSONA_FILE = "personas.json"


def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
    print(result)
    print(f"Spent a total of {cb.total_tokens} tokens\n")

    return result


if "OPENAI_API_KEY" not in os.environ:
    # set the environment variable to the contents of the file
    os.environ["OPENAI_API_KEY"] = open("openai_key.private").read().strip()

# Initialize the language model
llm_model_name = "gpt-3.5-turbo"
llm_temperature = 0.7
llm = ChatOpenAI(
    model_name=llm_model_name,
    verbose=False,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    temperature=llm_temperature,
)


# Initialize the summary model
summary_model_name = "gpt-3.5-turbo"
summary_temperature = 0.1
summary_llm = ChatOpenAI(model_name=summary_model_name, temperature=summary_temperature)

personas = {
    "financial manager": {
        "description": (
            "I am a skilled financial manager familiar with analysis and investing in"
            " equities, fixed income, alternative investments, and real estate. I am"
            " familiar with the use of derivatives and options to hedge risk and"
            " enhance returns. I am familiar with the use of leverage to enhance"
            " returns. I am familiar with the use of leverage to hedge risk. I am also"
            " an excellent mentor and when I use financial jargon, I will always"
            " provide a clear definition for the jargon terms at the end of my response"
        ),
        "extra_context": [],
    },
    "default": {
        "description": (
            "I am a highly intelligent question answering bot. If you ask me a"
            " question that is rooted in truth, I will give you the answer. If you"
            " ask me a question that isnonsense, trickery, or has no clear answer,"
            " I will respond with a nonsenseresponse."
        ),
        "extra_context": [],
    },
    "linux computer": {
        "description": (
            "Act as a Linux terminal. I will type commands and you will reply with what"
            " the terminal should show. Only reply with the terminal output inside one"
            " unique code block, and nothing else. Do not write explanations. Do not"
            " type commands unless I instruct you to do so. When I need to tell you"
            " something that is not a command I will do so by putting text inside"
            " square brackets [like this]."
        ),
        "extra_context": [
            "when you reply, put the entire reply in a single code block ```like"
            " this```"
        ],
    },
}

# if ther is an EXTRA_PERSONA_FILE, load the personas from there
if os.path.exists(EXTRA_PERSONA_FILE):
    import json

    with open(EXTRA_PERSONA_FILE) as f:
        extra_personas = json.load(f)
    personas.update(extra_personas)


memory = ConversationSummaryBufferMemory(
    llm=summary_llm, max_token_limit=1000, return_messages=True
)


# there is a bug in actually using templates with the memory object, so we
# build the prompt template manually


# build the prompt template; note: the MessagesPlaceholder is required
# to be able to access the history of messages, its variable "history" will be
# replaced with the history of messages by the conversation chain as provided by
# the memory object.
def build_prompt(persona):
    extra_prompts = []
    for extra in personas[persona]["extra_context"]:
        extra_prompts.append(HumanMessagePromptTemplate.from_template(extra))

    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(personas[persona]["description"]),
            *extra_prompts,
            # SystemMessage(content="You are a professor, act like one!"),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )


prompt = build_prompt("default")
conversation = ConversationChain(llm=llm, verbose=True, prompt=prompt, memory=memory)


mutli_line_input = False
while True:
    if mutli_line_input:
        print(
            "[Multi-line] Type 'quit' to exit, 'new' to start a new conversation, 'ml'"
            " to switch off multi-line input.  Press ctrl+d or enter '.' on a blank"
            " line: to process input."
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
            "Type 'quit' to exit, 'new' to start a new conversation, 'ml' to switch to"
            " multi-line input: "
        )

    if query == "quit":
        exit()
    if query == "new":
        print("Starting a new conversation...")
        memory.clear()
    elif query == "ml":
        mutli_line_input = not mutli_line_input

    # if query starts with 'persona', or 'personas' get the persona, if it no persona
    # is specified, show the available personas
    elif query == "personas" or query == "persona":
        print("Available personas:")
        for persona in personas:
            print(f" - {persona}")
    elif query.startswith("persona"):
        # load the new persona
        persona = query.split(maxsplit=1)[1].strip()
        if persona not in personas:
            print(f"Persona '{persona}' not found")
            continue
        print(f"Setting persona to '{persona}'")
        # have to rebuild prompt and chaine due to
        # https://github.com/hwchase17/langchain/issues/1800
        prompt = build_prompt(persona)
        conversation = ConversationChain(
            llm=llm, verbose=True, prompt=prompt, memory=memory
        )
        memory.clear()

    else:
        # below doesn't work yet due to an error in the framework
        # https://github.com/hwchase17/langchain/issues/1800
        # print(conversation({"input": query, "persona": personas["linux computer"]}))
        print(conversation.run(query))
        print("\n")
