import openai

# Set the OpenAI API key
openai.api_key = open("openai_key.txt").read()

#model = "gpt-3.5-turbo"
model = "gpt-4-0613"

# persona = "I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with a nonsense response."
# persona = "I am a skilled financial advisor who can help you with your financial planning. I can help you with your retirement planning, your investment planning, and your tax planning. I can also help you with your estate planning, your insurance planning, and your education planning. I can help you with your financial planning for your children, your grandchildren, and your great-grandchildren. I will answer all questions related to financial planning. If my answers include math, I will quietly check the math multiple times before replying ensuring the math is correct before answering"
persona = "I am a skilled financial asset manager familiar with analysis and investing in equities, fixed income, alternative investments, and real estate. I am familiar with the use of derivatives and options to hedge risk and enhance returns. I am familiar with the use of leverage to enhance returns. I am familiar with the use of leverage to hedge risk. I am also an excellent mentor and when I use financial jargon, I will always provide a clear definition for the jargon terms at the end of my response"
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
