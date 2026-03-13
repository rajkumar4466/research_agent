"""AI Research Agent — built with CrewAI + guardrails.

Uses:
- Custom tools: web_search (DuckDuckGo), calculator (safe math)
- Memory tools: record_fact, recall_facts (working memory)
- MCP tools: Playwright browser (filtered to navigate + snapshot only)
- Guardrails: search/browse budgets, dedup, MCP tool filtering
- LLM: OpenAI GPT-4o
"""

import sys

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters

from tools import CUSTOM_TOOLS
from guardrails import build_guarded_tools

load_dotenv()


AGENT_BACKSTORY = """\
You are a meticulous research agent that answers questions by searching the web and reading pages.

## Your workflow
1. THINK about what you need to find. Break the question into sub-questions.
2. Use web_search to discover relevant pages (titles, URLs, snippets).
3. Use browser_navigate to read the most promising pages for details.
4. After EVERY search or page read, call record_fact for each new fact you learn.
   This is critical — facts not recorded will be lost.
5. When you have enough facts OR a tool tells you a limit is reached,
   call recall_facts to review everything, then write your final answer.

## Rules you MUST follow
- NEVER fabricate information. Only state facts you actually found in sources.
- ALWAYS cite every factual claim with [Source: URL].
- If sources conflict, present both perspectives and note the disagreement.
- If you cannot find something, explicitly say so — do NOT guess or fill in gaps.
- Be efficient: don't repeat searches or revisit pages.
- When a tool returns a "limit reached" message, STOP using that tool immediately.
- Always call recall_facts before writing your final answer to ensure accuracy.

## Quality standards
- Prefer primary sources (official sites, research papers) over secondary ones.
- Include dates when information is time-sensitive.
- Distinguish between facts and opinions in your sources.
- Structure your answer with clear sections when the topic is complex.\
"""

TASK_DESCRIPTION = """\
Research and answer the following question:

{query}

Instructions:
1. Use web_search to find relevant pages first. Record facts from snippets immediately.
2. Use browser_navigate on the most relevant URLs to get details. Record new facts.
3. Use calculator for any math computations.
4. Call recall_facts before writing your final answer.
5. Cite EVERY factual claim with [Source: URL].
6. If sources conflict, mention both perspectives.
7. If you can't find something, say so — don't guess.
8. When a tool tells you a limit is reached, stop using that tool and write your answer.

IMPORTANT: Your answer must be based ONLY on facts you recorded with record_fact.
Call recall_facts to see all your verified facts, then synthesize your final answer.\
"""

EXPECTED_OUTPUT = """\
A comprehensive, well-structured answer where every factual claim is cited
with [Source: URL]. Use markdown formatting. If the question involves numbers
or comparisons, include the specific data points. Acknowledge any gaps or
conflicting information found during research.\
"""


def build_crew(query: str, mcp_tools: list) -> Crew:
    """Build a CrewAI crew for a research query."""

    guarded_tools, tracker = build_guarded_tools(CUSTOM_TOOLS, mcp_tools)

    print(f"  Tools: {[t.name for t in guarded_tools]}")
    print(f"  Budgets: {tracker.max_searches} searches, {tracker.max_browses} browses\n")

    llm = LLM(model="gpt-4o", temperature=0)

    def step_callback(step_output):
        """Monitor agent progress and detect stalling."""
        if tracker.check_stalling():
            print("\n⚠ No new facts in 3 steps — agent should synthesize soon.\n")

    researcher = Agent(
        role="Web Research Specialist",
        goal=(
            "Find accurate, up-to-date information(article needs to be the latest date) by searching the web "
            "and reading relevant pages. Record every fact with record_fact. "
            "Always call recall_facts before writing your final answer. "
            "Never make up information — every claim must come from a "
            "source you actually visited."
        ),
        backstory=AGENT_BACKSTORY,
        tools=guarded_tools,
        llm=llm,
        verbose=True,
        max_iter=15,
        allow_delegation=False,
        step_callback=step_callback,
    )

    research_task = Task(
        description=TASK_DESCRIPTION.format(query=query),
        expected_output=EXPECTED_OUTPUT,
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

    playwright_params = StdioServerParameters(
        command="npx",
        args=["@playwright/mcp@latest"],
    )

    with MCPServerAdapter(playwright_params) as mcp_tools:
        print(f"\nMCP tools available: {[t.name for t in mcp_tools]}")
        print(f"Custom tools: {[t.name for t in CUSTOM_TOOLS]}")

        if query:
            crew = build_crew(query, mcp_tools)
            result = crew.kickoff()
            print("\n" + "=" * 60)
            print("ANSWER")
            print("=" * 60)
            print(result.raw)
        else:
            print("\nAI Research Agent (CrewAI + Guardrails)")
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
