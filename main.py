"""CLI entry point for the AI Research Agent."""

import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from agent.loop import run_agent
from tools.mcp_client import MCPClient

load_dotenv()
console = Console()


async def main():
    console.print(
        Panel(
            "[bold]AI Research Agent[/bold]\n"
            "Ask me anything — I'll search the web, read pages, and synthesize an answer.\n"
            "Type [bold]quit[/bold] or [bold]exit[/bold] to stop.",
            border_style="blue",
        )
    )

    client = AsyncOpenAI()

    # Connect to MCP servers
    mcp = MCPClient()
    console.print("[dim]Connecting to MCP servers...[/dim]")
    await mcp.connect()
    console.print(f"[green]Connected! Available tools:[/green]")
    console.print(f"[dim]{mcp.describe_tools()}[/dim]")
    console.print(f"[dim]  - calculator: Evaluate math expressions[/dim]\n")

    try:
        while True:
            console.print()
            query = console.input("[bold blue]You:[/bold blue] ").strip()

            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                console.print("[dim]Goodbye![/dim]")
                break

            console.print()
            answer = await run_agent(query, client, mcp)
            console.print()
            console.print(
                Panel(Markdown(answer), title="[bold green]Answer[/bold green]", border_style="green")
            )
    finally:
        await mcp.close()


if __name__ == "__main__":
    asyncio.run(main())
