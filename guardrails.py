"""Guardrails that wrap tools with budget limits, dedup, fact tracking, and filtering.

CrewAI doesn't provide these natively, so we wrap each tool's _run method
with checks before the actual execution happens.
"""

from dataclasses import dataclass, field

from crewai.tools import tool


# ── Fact tracking ──────────────────────────────────────────────────────────

@dataclass
class Fact:
    claim: str
    source_url: str


# ── Budget tracker (shared state across all tool calls in a query) ──────────

class BudgetTracker:
    """Tracks usage, facts, and budgets across all tools for a single query."""

    def __init__(
        self,
        max_searches: int = 5,
        max_browses: int = 8,
    ):
        self.max_searches = max_searches
        self.max_browses = max_browses
        self.search_queries: list[str] = []
        self.urls_visited: set[str] = set()
        self.facts: list[Fact] = []
        self.steps_with_no_new_facts: int = 0
        self.last_fact_count: int = 0

    def check_search(self, query: str) -> str | None:
        if query.lower() in [q.lower() for q in self.search_queries]:
            return f"You already searched for '{query}'. Try a different query."
        if len(self.search_queries) >= self.max_searches:
            return (
                f"Search limit reached ({self.max_searches}). "
                "STOP searching. Use recall_facts and write your final answer NOW."
            )
        return None

    def record_search(self, query: str):
        self.search_queries.append(query)

    def check_browse(self, url: str) -> str | None:
        if url in self.urls_visited:
            return f"Already visited {url}. Use the information you already collected."
        if len(self.urls_visited) >= self.max_browses:
            return (
                f"Browse limit reached ({self.max_browses}). "
                "STOP browsing. Use recall_facts and write your final answer NOW."
            )
        return None

    def record_browse(self, url: str):
        self.urls_visited.add(url)

    def add_fact(self, claim: str, source_url: str):
        self.facts.append(Fact(claim=claim, source_url=source_url))

    def get_facts_summary(self) -> str:
        if not self.facts:
            return "No facts recorded yet."
        lines = []
        for i, f in enumerate(self.facts, 1):
            lines.append(f"{i}. {f.claim} [Source: {f.source_url}]")
        return "\n".join(lines)

    def get_status(self) -> str:
        return (
            f"Searches: {len(self.search_queries)}/{self.max_searches} | "
            f"Pages: {len(self.urls_visited)}/{self.max_browses} | "
            f"Facts: {len(self.facts)}"
        )

    def check_stalling(self) -> bool:
        """Check if the agent is making no progress. Call after each step."""
        if len(self.facts) == self.last_fact_count:
            self.steps_with_no_new_facts += 1
        else:
            self.steps_with_no_new_facts = 0
        self.last_fact_count = len(self.facts)
        return self.steps_with_no_new_facts >= 3


# ── Memory tools (record & recall facts) ──────────────────────────────────

def make_record_fact_tool(tracker: BudgetTracker):
    """Create a tool that lets the agent record verified facts."""

    @tool("record_fact")
    def record_fact(claim: str, source_url: str) -> str:
        """Record a verified fact with its source URL. You MUST call this
        every time you learn a new fact from a search result or a page.
        This is your working memory — facts not recorded here will be lost.
        Args:
            claim: The factual claim (e.g., 'Python 3.12 was released on Oct 2, 2023')
            source_url: The URL where this fact was found"""
        tracker.add_fact(claim, source_url)
        return f"Fact recorded. Total facts: {len(tracker.facts)}. {tracker.get_status()}"

    return record_fact


def make_recall_facts_tool(tracker: BudgetTracker):
    """Create a tool that lets the agent review all gathered facts."""

    @tool("recall_facts")
    def recall_facts() -> str:
        """Retrieve all facts you have recorded so far. Call this BEFORE
        writing your final answer to make sure you use only verified facts.
        Returns all recorded facts with their source URLs."""
        summary = tracker.get_facts_summary()
        return f"=== YOUR VERIFIED FACTS ===\n{summary}\n\n{tracker.get_status()}"

    return recall_facts


# ── Guarded tool factories ─────────────────────────────────────────────────

# Max characters to return from tool outputs to avoid blowing up LLM context
MAX_SEARCH_RESULT_CHARS = 3000
MAX_BROWSE_RESULT_CHARS = 5000


def _truncate(text: str, limit: int) -> str:
    """Truncate text to limit, appending a notice if trimmed."""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n[... truncated — record important facts now before they are lost]"


def make_guarded_search(original_tool, tracker: BudgetTracker):
    """Wrap the search tool with dedup + budget guardrails."""

    @tool("web_search")
    def guarded_web_search(query: str) -> str:
        """Search the web using DuckDuckGo. Use this to find relevant pages,
        articles, and information. Returns titles, URLs, and snippets.
        Always use this before browsing to discover relevant URLs first."""
        blocked = tracker.check_search(query)
        if blocked:
            return blocked
        tracker.record_search(query)
        result = original_tool.run(query=query)
        return _truncate(result, MAX_SEARCH_RESULT_CHARS)

    return guarded_web_search


def make_guarded_browser(original_tool, tracker: BudgetTracker):
    """Wrap browser_navigate with URL dedup + budget guardrails."""

    @tool("browser_navigate")
    def guarded_browser_navigate(url: str) -> str:
        """Navigate to a URL and read the page content. Use this after
        web_search to read specific pages in detail. Opens a real browser
        that handles JavaScript-rendered pages."""
        blocked = tracker.check_browse(url)
        if blocked:
            return blocked
        tracker.record_browse(url)
        result = original_tool.run(url=url)
        return _truncate(result, MAX_BROWSE_RESULT_CHARS)

    return guarded_browser_navigate


# ── MCP tool filtering ─────────────────────────────────────────────────────

ALLOWED_MCP_TOOLS = {"browser_navigate", "browser_snapshot"}


def filter_mcp_tools(mcp_tools: list) -> list:
    """Keep only the MCP tools the research agent actually needs."""
    return [t for t in mcp_tools if t.name in ALLOWED_MCP_TOOLS]


# ── Build guarded tool set ─────────────────────────────────────────────────

def build_guarded_tools(custom_tools: list, mcp_tools: list) -> tuple[list, BudgetTracker]:
    """Build the final tool list with all guardrails applied.

    Returns (tools, tracker) so the caller can inspect budget state if needed.
    """
    tracker = BudgetTracker()

    filtered_mcp = filter_mcp_tools(mcp_tools)

    guarded_tools = []

    # Wrap search tool
    for t in custom_tools:
        if t.name == "web_search":
            guarded_tools.append(make_guarded_search(t, tracker))
        else:
            guarded_tools.append(t)

    # Wrap browser_navigate, pass through browser_snapshot
    for t in filtered_mcp:
        if t.name == "browser_navigate":
            guarded_tools.append(make_guarded_browser(t, tracker))
        else:
            guarded_tools.append(t)

    # Add memory tools
    guarded_tools.append(make_record_fact_tool(tracker))
    guarded_tools.append(make_recall_facts_tool(tracker))

    return guarded_tools, tracker
