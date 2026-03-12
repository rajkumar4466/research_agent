"""Working memory (scratchpad) for the agent.

Tracks facts gathered, sources visited, and remaining budget
so the agent knows where it stands at every step.
"""

from dataclasses import dataclass, field


@dataclass
class Fact:
    claim: str
    source_url: str


@dataclass
class WorkingMemory:
    query: str
    facts: list[Fact] = field(default_factory=list)
    urls_visited: set[str] = field(default_factory=set)
    search_queries: list[str] = field(default_factory=list)
    steps_taken: int = 0

    # Guardrails
    max_steps: int = 15
    max_searches: int = 5
    max_browses: int = 8

    def add_fact(self, claim: str, source_url: str):
        self.facts.append(Fact(claim=claim, source_url=source_url))

    def mark_url_visited(self, url: str):
        self.urls_visited.add(url)

    def record_search(self, query: str):
        self.search_queries.append(query)

    def is_duplicate_search(self, query: str) -> bool:
        return query.lower() in [q.lower() for q in self.search_queries]

    def is_url_visited(self, url: str) -> bool:
        return url in self.urls_visited

    def has_budget(self) -> bool:
        return self.steps_taken < self.max_steps

    def can_search(self) -> bool:
        return len(self.search_queries) < self.max_searches

    def can_browse(self) -> bool:
        return len(self.urls_visited) < self.max_browses

    def increment_step(self):
        self.steps_taken += 1

    def summary(self) -> str:
        """Return a status string for the agent to understand its state."""
        lines = [
            f"Query: {self.query}",
            f"Steps: {self.steps_taken}/{self.max_steps}",
            f"Searches: {len(self.search_queries)}/{self.max_searches}",
            f"Pages browsed: {len(self.urls_visited)}/{self.max_browses}",
            f"Facts collected: {len(self.facts)}",
        ]
        if self.facts:
            lines.append("\nFacts so far:")
            for i, fact in enumerate(self.facts, 1):
                lines.append(f"  {i}. {fact.claim} [source: {fact.source_url}]")
        return "\n".join(lines)
