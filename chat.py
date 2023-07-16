import openai
import os

# Set the OpenAI API key if the environment variable OPENAI_API_KEY exists, otherwise
# load it from the file openai_key.private

if "OPENAI_API_KEY" in os.environ:
    openai.api_key = os.environ["OPENAI_API_KEY"]
else:
    openai.api_key = open("openai_key.private").read()

# Set the file to load extra personas from
EXTRA_PERSONA_FILE = "personas.json"


# Set the model to use  - see
# https://beta.openai.com/docs/api-reference/create-completion for more options
# on the model parameter

model = "gpt-3.5-turbo"
# model = "gpt-4"
# model = "gpt-4-0613"

# dictionary of persona strings
personas = {
    "financial manager": (
        "I am a skilled financial manager familiar with analysis and investing in"
        " equities, fixed income, alternative investments, and real estate. I am"
        " familiar with the use of derivatives and options to hedge risk and enhance"
        " returns. I am familiar with the use of leverage to enhance returns. I am"
        " familiar with the use of leverage to hedge risk. I am also an excellent"
        " mentor and when I use financial jargon, I will always provide a clear"
        " definition for the jargon terms at the end of my response"
    ),
    "default": (
        "I am a highly intelligent question answering bot. If you ask me a question"
        " that is rooted in truth, I will give you the answer. If you ask me a question"
        " that isnonsense, trickery, or has no clear answer, I will respond with a"
        " nonsenseresponse."
    ),
}

# if ther is an EXTRA_PERSONA_FILE, load the personas from there
if os.path.exists(EXTRA_PERSONA_FILE):
    import json

    with open(EXTRA_PERSONA_FILE) as f:
        extra_personas = json.load(f)
    personas.update(extra_personas)

persona = personas["default"]

messages = []
messages.append({"role": "system", "content": persona})

while True:
    # Get the user's input
    prompt = input(
        "Ask away, or type 'quit' to exit, or 'new' to start a new conversation: "
    )

    if prompt == "quit":
        break

    if prompt == "new":
        messages = []
        messages.append({"role": "system", "content": persona})
        continue

    # Add the user's question to the prompt
    messages.append({"role": "user", "content": prompt})
    # print("{}\n".format(messages))
    print("\n" * 2)

    # Send the prompt to the ChatGPT model and get the response, maintain each prompt
    # and response and to send it back to the model as the conversation continues.

    # create variables to collect the stream of chunks
    collected_chunks = []
    collected_messages = []
    for chunk in openai.ChatCompletion.create(
        model=model, messages=messages, stream=True
    ):
        collected_chunks.append(chunk)  # save the event response
        try:
            chunk_message = chunk["choices"][0]["delta"]["content"]
        except KeyError:
            stop_reason = chunk["choices"][0]["finish_reason"]
        else:
            collected_messages.append(chunk_message)  # save the message
            print(chunk_message, end="", flush=True)  # print the message

    # join the list of messages into a single string
    response = "".join(collected_messages)

    messages.append({"role": "assistant", "content": response})

    print("\n" * 2)
