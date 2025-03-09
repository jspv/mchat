#!/usr/bin/env python3
import click
from textual_serve.server import Server


@click.command()
@click.option("--command", default="python -m mchat.mchat", help="Command to run")
@click.option("--port", default=5500, help="Port to serve on")
def serve(command, port):
    server = Server(command=command, port=port)
    server.serve()


if __name__ == "__main__":
    serve()
