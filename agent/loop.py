"""The ReAct agent loop — Reason, Act, Observe."""

import json

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from agent.memory import WorkingMemory
from agent.prompts import SYSTEM_PROMPT, SYNTHESIS_PROMPT
from tools import ALL_TOOLS

console = Console()

# Build lookup: tool_name -> tool instance
TOOL_MAP = {tool.name: tool for tool in ALL_TOOLS}


def _build_system_message(memory: WorkingMemory) -> str:
    return SYSTEM_PROMPT.format(memory_status=memory.summary())


def _stream_response(client: OpenAI, messages: list, tools: list) -> dict:
    """Stream an OpenAI response, printing text chunks in real-time.

    Returns the fully assembled message (with tool_calls if any).
    """
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        stream=True,
    )

    # Accumulators
    content_chunks = []
    tool_calls_data = {}  # index -> {id, name, arguments}

    for chunk in stream:
        delta = chunk.choices[0].delta

        # Stream text content
        if delta.content:
            console.print(delta.content, end="", style="dim")
            content_chunks.append(delta.content)

        # Accumulate tool calls
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls_data:
                    tool_calls_data[idx] = {
                        "id": tc.id or "",
                        "name": tc.function.name or "" if tc.function else "",
                        "arguments": "",
                    }
                if tc.id:
                    tool_calls_data[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        tool_calls_data[idx]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_calls_data[idx]["arguments"] += tc.function.arguments

    if content_chunks:
        console.print()  # newline after streamed text

    # Build the assembled message
    assembled = {"role": "assistant", "content": "".join(content_chunks) or None}

    if tool_calls_data:
        assembled["tool_calls"] = [
            {
                "id": data["id"],
                "type": "function",
                "function": {
                    "name": data["name"],
                    "arguments": data["arguments"],
                },
            }
            for data in sorted(tool_calls_data.items())
            for data in [data[1]]
        ]

    return assembled


def _execute_tool(tool_name: str, arguments: dict, memory: WorkingMemory) -> str:
    """Execute a tool and apply guardrails."""
    tool = TOOL_MAP.get(tool_name)
    if not tool:
        return f"Unknown tool: {tool_name}"

    # Guardrails
    if tool_name == "web_search":
        query = arguments.get("query", "")
        if memory.is_duplicate_search(query):
            return f"You already searched for '{query}'. Use existing results or try a different query."
        if not memory.can_search():
            return f"Search limit reached ({memory.max_searches}). Synthesize with what you have."
        memory.record_search(query)

    if tool_name == "browse_webpage":
        url = arguments.get("url", "")
        if memory.is_url_visited(url):
            return f"Already visited {url}. Use the facts you already collected."
        if not memory.can_browse():
            return f"Browse limit reached ({memory.max_browses}). Synthesize with what you have."
        memory.mark_url_visited(url)

    return tool.run(**arguments)


def run_agent(query: str, client: OpenAI) -> str:
    """Run the ReAct agent loop and return the final answer."""
    memory = WorkingMemory(query=query)
    tool_specs = [tool.to_openai_spec() for tool in ALL_TOOLS]

    messages = [
        {"role": "system", "content": _build_system_message(memory)},
        {"role": "user", "content": query},
    ]

    steps_with_no_new_facts = 0

    while memory.has_budget():
        memory.increment_step()

        # Update system message with latest memory state
        messages[0]["content"] = _build_system_message(memory)

        console.print(
            Panel(
                f"Step {memory.steps_taken}/{memory.max_steps} | "
                f"Facts: {len(memory.facts)} | "
                f"Searches: {len(memory.search_queries)}/{memory.max_searches} | "
                f"Pages: {len(memory.urls_visited)}/{memory.max_browses}",
                title="[bold cyan]Agent Status[/bold cyan]",
                border_style="cyan",
            )
        )

        # Get LLM response (streamed)
        assistant_msg = _stream_response(client, messages, tool_specs)
        messages.append(assistant_msg)

        # If no tool calls, the agent is done — this is the final answer
        if "tool_calls" not in assistant_msg or not assistant_msg.get("tool_calls"):
            console.print("\n[bold green]Agent finished.[/bold green]\n")
            return assistant_msg.get("content", "No answer generated.")

        # Execute each tool call
        fact_count_before = len(memory.facts)

        for tool_call in assistant_msg["tool_calls"]:
            fn_name = tool_call["function"]["name"]
            try:
                fn_args = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError:
                fn_args = {}

            console.print(f"\n[bold yellow]Tool:[/bold yellow] {fn_name}")
            console.print(f"[yellow]Args:[/yellow] {json.dumps(fn_args, indent=2)}")

            result = _execute_tool(fn_name, fn_args, memory)

            # Show truncated result
            display_result = result[:500] + "..." if len(result) > 500 else result
            console.print(Panel(display_result, title=f"[yellow]{fn_name} result[/yellow]", border_style="yellow"))

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": result,
            })

        # Diminishing returns detection
        if len(memory.facts) == fact_count_before:
            steps_with_no_new_facts += 1
        else:
            steps_with_no_new_facts = 0

        if steps_with_no_new_facts >= 3:
            console.print("[bold red]No new facts in 3 steps. Forcing synthesis.[/bold red]")
            messages.append({
                "role": "user",
                "content": SYNTHESIS_PROMPT,
            })

    # Budget exhausted — force final answer
    console.print("[bold red]Step budget exhausted. Synthesizing with available facts.[/bold red]")
    messages.append({
        "role": "user",
        "content": SYNTHESIS_PROMPT,
    })
    final_msg = _stream_response(client, messages, tools=[])
    return final_msg.get("content", "No answer generated.")
