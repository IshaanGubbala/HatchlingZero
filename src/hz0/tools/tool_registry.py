"""Tool registry for HZ v2.0 tool-use training.

Defines canonical tool schemas and examples.
"""

import json
from typing import Dict, List, Any


class ToolDefinition:
    """Single tool schema + examples."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        examples: List[Dict[str, Any]] = None,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.examples = examples or []

    def to_dict(self) -> Dict:
        """Export as JSON schema."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def validate_call(self, arguments: Dict) -> bool:
        """Check if call matches schema."""
        required = self.parameters.get("required", [])
        props = self.parameters.get("properties", {})

        for req in required:
            if req not in arguments:
                return False

        for key, value in arguments.items():
            if key not in props:
                return False

        return True


class ToolRegistry:
    """Registry of available tools for v2.0."""

    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}
        self._register_core_tools()

    def _register_core_tools(self):
        """Register initial tool set (v2.0)."""

        # 1. File operations
        self.register(
            ToolDefinition(
                name="read_file",
                description="Read contents of a UTF-8 text file",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                    },
                    "required": ["path"],
                },
            )
        )

        self.register(
            ToolDefinition(
                name="write_file",
                description="Write text to a file",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                        "contents": {"type": "string", "description": "File contents"},
                    },
                    "required": ["path", "contents"],
                },
            )
        )

        # 2. JSON operations
        self.register(
            ToolDefinition(
                name="parse_json",
                description="Parse and validate JSON string",
                parameters={
                    "type": "object",
                    "properties": {
                        "json_str": {"type": "string", "description": "JSON string"},
                    },
                    "required": ["json_str"],
                },
            )
        )

        # 3. Arithmetic
        self.register(
            ToolDefinition(
                name="calculator",
                description="Perform arithmetic calculation",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression (e.g., '2+2')",
                        },
                    },
                    "required": ["expression"],
                },
            )
        )

        # 4. Web search (simulate)
        self.register(
            ToolDefinition(
                name="search",
                description="Search for information",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            )
        )

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition:
        """Get tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> List[Dict]:
        """Get all tools as JSON schemas."""
        return [tool.to_dict() for tool in self.tools.values()]

    def to_json(self) -> str:
        """Export all tools as JSON array."""
        return json.dumps(self.list_tools(), indent=2)


# Global registry
REGISTRY = ToolRegistry()
