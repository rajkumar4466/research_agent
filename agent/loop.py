"""The ReAct agent loop — Reason, Act, Observe.

Uses MCP servers for search/browse and custom tools for calculator.
"""

import json

from openai import AsyncOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from agent.memory import WorkingMemory
from agent.prompts import SYSTEM_PROMPT, SYNTHESIS_PROMPT
from tools import CUSTOM_TOOLS
from tools.mcp_client import MCPClient

console = Console()

# Custom tool lookup
CUSTOM_TOOL_MAP = {tool.name: tool for tool in CUSTOM_TOOLS}


def _build_system_message(memory: WorkingMemory, mcp_tool_descriptions: str) -> str:
    return SYSTEM_PROMPT.format(
        memory_status=memory.summary(),
        available_tools=mcp_tool_descriptions,
    )


async def _stream_response(client: AsyncOpenAI, messages: list, tools: list) -> dict:
    """Stream an OpenAI response, printing text chunks in real-time."""
    kwargs = {
        "model": "gpt-4o",
        "messages": messages,
        "stream": True,
    }
    if tools:
        kwargs["tools"] = tools

    stream = await client.chat.completions.create(**kwargs)

    content_chunks = []
    tool_calls_data = {}

    async for chunk in stream:
        delta = chunk.choices[0].delta

        if delta.content:
            console.print(delta.content, end="", style="dim")
            content_chunks.append(delta.content)

        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls_data:
                    tool_calls_data[idx] = {"id": "", "name": "", "arguments": ""}
                if tc.id:
                    tool_calls_data[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        tool_calls_data[idx]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_calls_data[idx]["arguments"] += tc.function.arguments

    if content_chunks:
        console.print()

    assembled = {"role": "assistant", "content": "".join(content_chunks) or None}

    if tool_calls_data:
        assembled["tool_calls"] = [
            {
                "id": data["id"],
                "type": "function",
                "function": {"name": data["name"], "arguments": data["arguments"]},
            }
            for _, data in sorted(tool_calls_data.items())
        ]

    return assembled


async def _execute_tool(
    tool_name: str, arguments: dict, memory: WorkingMemory, mcp: MCPClient
) -> str:
    """Execute a tool — route to MCP server or custom tool."""

    # Guardrails — check both MCP and custom search/browse tools
    is_search = mcp.is_search_tool(tool_name) or tool_name == "web_search"
    is_browse = mcp.is_browse_tool(tool_name)

    if is_search:
        query = arguments.get("query", "")
        if memory.is_duplicate_search(query):
            return f"You already searched for '{query}'. Try a different query."
        if not memory.can_search():
            return f"Search limit reached ({memory.max_searches}). Synthesize your answer now."
        memory.record_search(query)

    if is_browse:
        url = arguments.get("url", "")
        if url and memory.is_url_visited(url):
            return f"Already visited {url}. Use existing facts."
        if not memory.can_browse():
            return f"Browse limit reached ({memory.max_browses}). Synthesize your answer now."
        if url:
            memory.mark_url_visited(url)

    # Route to MCP or custom
    if mcp.has_tool(tool_name):
        return await mcp.call_tool(tool_name, arguments)
    elif tool_name in CUSTOM_TOOL_MAP:
        return CUSTOM_TOOL_MAP[tool_name].run(**arguments)
    else:
        return f"Unknown tool: {tool_name}"


async def run_agent(query: str, client: AsyncOpenAI, mcp: MCPClient) -> str:
    """Run the ReAct agent loop and return the final answer."""
    memory = WorkingMemory(query=query)

    # Combine MCP tools + custom tools into OpenAI format
    tool_specs = mcp.get_openai_tool_specs() + [t.to_openai_spec() for t in CUSTOM_TOOLS]

    mcp_descriptions = mcp.describe_tools()
    custom_descriptions = "\n".join(
        f"  - {t.name}: {t.description}" for t in CUSTOM_TOOLS
    )
    all_tool_descriptions = mcp_descriptions + "\n" + custom_descriptions

    messages = [
        {"role": "system", "content": _build_system_message(memory, all_tool_descriptions)},
        {"role": "user", "content": query},
    ]

    steps_with_no_new_facts = 0

    while memory.has_budget():
        memory.increment_step()
        messages[0]["content"] = _build_system_message(memory, all_tool_descriptions)

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

        assistant_msg = await _stream_response(client, messages, tool_specs)
        messages.append(assistant_msg)

        # No tool calls → agent is done
        if not assistant_msg.get("tool_calls"):
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

            result = await _execute_tool(fn_name, fn_args, memory, mcp)

            display_result = result[:500] + "..." if len(result) > 500 else result
            console.print(
                Panel(display_result, title=f"[yellow]{fn_name} result[/yellow]", border_style="yellow")
            )

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
            messages.append({"role": "user", "content": SYNTHESIS_PROMPT})

    # Budget exhausted
    console.print("[bold red]Step budget exhausted. Synthesizing with available facts.[/bold red]")
    messages.append({"role": "user", "content": SYNTHESIS_PROMPT})
    final_msg = await _stream_response(client, messages, tools=[])
    return final_msg.get("content", "No answer generated.")
