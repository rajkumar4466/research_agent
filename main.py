"""AI Research Agent — built with CrewAI.

Uses:
- Custom tools: web_search (DuckDuckGo), calculator (safe math)
- MCP tools: Playwright browser (real Chromium for JS-heavy pages)
- LLM: OpenAI GPT-4o
"""

import sys

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters

from tools import CUSTOM_TOOLS

load_dotenv()


def build_crew(query: str, mcp_tools: list) -> Crew:
    """Build a CrewAI crew for a research query."""

    all_tools = CUSTOM_TOOLS + mcp_tools

    llm = LLM(model="gpt-4o-mini", temperature=0)

    researcher = Agent(
        role="Web Research Specialist",
        goal=(
            "Find accurate, up-to-date information by searching the web "
            "and reading relevant pages. Always cite sources with URLs."
        ),
        backstory=(
            "You are an expert researcher who methodically searches for information, "
            "reads primary sources, and extracts verified facts. You never make up "
            "information — every claim must come from a source you actually visited. "
            "You use web_search first to discover URLs, then browser_navigate to read "
            "pages in detail. You use the calculator for any math."
        ),
        tools=all_tools,
        llm=llm,
        verbose=True,
        max_iter=15,
        allow_delegation=False,
    )

    research_task = Task(
        description=(
            f"Research and answer the following question:\n\n"
            f"{query}\n\n"
            f"Instructions:\n"
            f"1. Use web_search to find relevant pages first.\n"
            f"2. Use browser_navigate to read the most relevant pages for details.\n"
            f"3. Use calculator for any math computations.\n"
            f"4. Cite every fact with its source URL.\n"
            f"5. If sources conflict, mention both perspectives.\n"
            f"6. If you can't find something, say so — don't guess."
        ),
        expected_output=(
            "A comprehensive, well-structured answer with inline source citations "
            "[Source: URL] for every factual claim. Use markdown formatting."
        ),
        agent=researcher,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        verbose=True,
    )

    return crew


def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None

    # Connect to Playwright MCP server for browser tools
    playwright_params = StdioServerParameters(
        command="npx",
        args=["@playwright/mcp@latest"],
    )

    with MCPServerAdapter(playwright_params) as mcp_tools:
        print(f"\nMCP tools loaded: {[t.name for t in mcp_tools]}")
        print(f"Custom tools: {[t.name for t in CUSTOM_TOOLS]}\n")

        if query:
            # Single query mode (from command line args)
            crew = build_crew(query, mcp_tools)
            result = crew.kickoff()
            print("\n" + "=" * 60)
            print("ANSWER")
            print("=" * 60)
            print(result.raw)
        else:
            # Interactive mode
            print("AI Research Agent (CrewAI)")
            print("Type your question, or 'quit' to exit.\n")

            while True:
                query = input("You: ").strip()
                if not query:
                    continue
                if query.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break

                crew = build_crew(query, mcp_tools)
                result = crew.kickoff()
                print("\n" + "=" * 60)
                print("ANSWER")
                print("=" * 60)
                print(result.raw)
                print()


if __name__ == "__main__":
    main()
