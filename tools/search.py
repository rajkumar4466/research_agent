"""Custom search tool using DuckDuckGo (ddgs library)."""

from crewai.tools import tool
from ddgs import DDGS


@tool("web_search")
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo. Use this to find relevant pages,
    articles, and information. Returns titles, URLs, and snippets.
    Always use this before browsing to discover relevant URLs first."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return "No results found."

        output = []
        for i, r in enumerate(results, 1):
            output.append(
                f"{i}. {r['title']}\n"
                f"   URL: {r['href']}\n"
                f"   {r['body']}\n"
            )
        return "\n".join(output)

    except Exception as e:
        return f"Search failed: {e}"
