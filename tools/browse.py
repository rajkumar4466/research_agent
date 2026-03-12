import httpx
from bs4 import BeautifulSoup

from tools.base import Tool

MAX_CONTENT_LENGTH = 4000  # Truncate to control token usage


class BrowseTool(Tool):
    name = "browse_webpage"
    description = (
        "Fetch and read the content of a webpage. Returns the main text content "
        "of the page (HTML tags stripped). Use this after searching to read "
        "specific pages in detail."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The full URL of the webpage to read.",
            },
        },
        "required": ["url"],
    }

    def run(self, url: str) -> str:
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            }
            response = httpx.get(url, headers=headers, timeout=15, follow_redirects=True)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script, style, nav, footer elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)

            # Clean up excessive whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = "\n".join(lines)

            if len(text) > MAX_CONTENT_LENGTH:
                text = text[:MAX_CONTENT_LENGTH] + "\n\n[... content truncated]"

            if not text:
                return "Could not extract text content from this page."

            return f"Content from {url}:\n\n{text}"

        except httpx.HTTPStatusError as e:
            return f"HTTP error {e.response.status_code} fetching {url}"
        except Exception as e:
            return f"Failed to browse {url}: {e}"
