from abc import ABC, abstractmethod


class Tool(ABC):
    """Base class for all agent tools."""

    name: str
    description: str
    parameters: dict

    @abstractmethod
    def run(self, **kwargs) -> str:
        """Execute the tool and return a string result."""
        pass

    def to_openai_spec(self) -> dict:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
