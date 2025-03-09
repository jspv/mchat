from mchat.mchatweb import WebChatApp

app = WebChatApp()


if __name__ in {"__main__", "__mp_main__"}:
    app.run(
        port=8881,
        title="MChat - Multi-Model Chat Framework",
        favicon="static/favicon-32x32.png",
        dark=True,
    )
