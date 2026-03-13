from ddgs import DDGS

from tools.base import Tool


class SearchTool(Tool):
    name = "web_search"
    description = (
        "Search the web using DuckDuckGo. Use this to find relevant pages, "
        "articles, and information. Returns titles, URLs, and snippets."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up.",
            },
            "max_results": {
                "type": "integer",
                "description": "Number of results to return (default 5, max 10).",
            },
        },
        "required": ["query"],
    }

    def run(self, query: str, max_results: int = 5) -> str:
        max_results = min(max_results, 10)
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))

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
