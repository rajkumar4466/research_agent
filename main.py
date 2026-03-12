"""CLI entry point for the AI Research Agent."""

import sys

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from agent.loop import run_agent

load_dotenv()
console = Console()


def main():
    console.print(
        Panel(
            "[bold]AI Research Agent[/bold]\n"
            "Ask me anything — I'll search the web, read pages, and synthesize an answer.\n"
            "Type [bold]quit[/bold] or [bold]exit[/bold] to stop.",
            border_style="blue",
        )
    )

    client = OpenAI()

    while True:
        console.print()
        query = console.input("[bold blue]You:[/bold blue] ").strip()

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        console.print()
        answer = run_agent(query, client)
        console.print()
        console.print(Panel(Markdown(answer), title="[bold green]Answer[/bold green]", border_style="green"))


if __name__ == "__main__":
    main()
