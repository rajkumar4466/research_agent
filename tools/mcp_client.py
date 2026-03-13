"""MCP client that connects to external MCP servers (search, browse).

Handles:
- Starting MCP servers as subprocesses (stdio transport)
- Discovering available tools from each server
- Filtering to only the tools we actually need
- Converting MCP tool schemas to OpenAI function calling format
- Executing tool calls and returning results
"""

from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# MCP servers we connect to
MCP_SERVERS = {
    "playwright": StdioServerParameters(
        command="npx",
        args=["@playwright/mcp@latest"],
    ),
}

# Only expose these tools to the LLM.
# Playwright has 15+ tools (drag, hover, screenshot, etc.) that would confuse
# the agent. We only need navigation + snapshot for reading pages.
ALLOWED_TOOLS = {
    "playwright": {"browser_navigate", "browser_snapshot"},
}

# Tools that count as "browsing" (for guardrail budget tracking).
# Exact names — no substring guessing.
BROWSE_TOOLS = {"browser_navigate"}

# Tools that count as "searching" (for guardrail budget tracking).
# Search is now a custom tool, not MCP — but keep this for reference.
SEARCH_TOOLS = {"web_search"}


class MCPClient:
    """Manages connections to multiple MCP servers."""

    def __init__(self):
        self._exit_stack = AsyncExitStack()
        # tool_name -> (session, server_name)
        self._tool_sessions: dict[str, tuple[ClientSession, str]] = {}
        # tool_name -> MCP tool definition
        self._tool_definitions: dict[str, object] = {}

    async def connect(self):
        """Connect to all MCP servers and discover their tools."""
        for server_name, params in MCP_SERVERS.items():
            try:
                read, write = await self._exit_stack.enter_async_context(
                    stdio_client(params)
                )
                session = await self._exit_stack.enter_async_context(
                    ClientSession(read, write)
                )
                await session.initialize()

                # Discover tools, but only keep the ones we allow
                allowed = ALLOWED_TOOLS.get(server_name)
                tools_result = await session.list_tools()
                for tool in tools_result.tools:
                    if allowed and tool.name not in allowed:
                        continue
                    self._tool_sessions[tool.name] = (session, server_name)
                    self._tool_definitions[tool.name] = tool

            except Exception as e:
                print(f"[Warning] Failed to connect to {server_name} MCP server: {e}")

    async def close(self):
        """Clean up all connections."""
        await self._exit_stack.aclose()

    def get_openai_tool_specs(self) -> list[dict]:
        """Convert all MCP tools to OpenAI function calling format."""
        specs = []
        for tool_name, tool_def in self._tool_definitions.items():
            specs.append({
                "type": "function",
                "function": {
                    "name": tool_def.name,
                    "description": tool_def.description or "",
                    "parameters": tool_def.inputSchema,
                },
            })
        return specs

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._tool_sessions

    def is_search_tool(self, tool_name: str) -> bool:
        return tool_name in SEARCH_TOOLS

    def is_browse_tool(self, tool_name: str) -> bool:
        return tool_name in BROWSE_TOOLS

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool on its MCP server and return the result as a string."""
        if tool_name not in self._tool_sessions:
            return f"Unknown MCP tool: {tool_name}"

        session, server_name = self._tool_sessions[tool_name]
        try:
            result = await session.call_tool(tool_name, arguments)
            # Extract text from result content
            if result.content:
                parts = []
                for block in result.content:
                    if hasattr(block, "text"):
                        parts.append(block.text)
                    else:
                        parts.append(str(block))
                return "\n".join(parts)
            return "Tool returned no content."

        except Exception as e:
            return f"Error calling {tool_name} on {server_name}: {e}"

    def list_tools(self) -> list[str]:
        """List all available tool names."""
        return list(self._tool_definitions.keys())

    def describe_tools(self) -> str:
        """Human-readable summary of available tools."""
        lines = []
        for name, tool_def in self._tool_definitions.items():
            _, server = self._tool_sessions[name]
            lines.append(f"  - {name} (from {server}): {tool_def.description}")
        return "\n".join(lines)
