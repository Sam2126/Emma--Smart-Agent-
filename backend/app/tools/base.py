"""
Minimal tool base class for the agent engine.

Replaces crewai.tools.BaseTool. A tool declares `name`, `description` and a
Pydantic `args_schema`, and implements `_run(**args)`. The engine validates
model-supplied arguments with the schema, calls the tool, and advertises it to
the LLM through `to_openai_tool()`.

The JSON schema sent to the model lists as required ONLY the arguments that
have no default. CrewAI's strict-mode conversion marked every argument
required, which made Groq reject perfectly valid calls such as
send_keys(window_hint='WhatsApp', keys='^f') with "missing properties: 'text',
'delay_ms'" — see utils/litellm_patch._relax_tool_schemas.
"""

from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel, ConfigDict, ValidationError


class BaseTool(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    args_schema: Type[BaseModel]

    def _run(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def run(self, **kwargs: Any) -> str:
        """Validate arguments against `args_schema`, then execute the tool."""
        try:
            validated = self.args_schema(**kwargs)
        except ValidationError as e:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or 'arguments'}: {err['msg']}"
                for err in e.errors()
            )
            return f"Invalid arguments for {self.name}: {problems}"
        result = self._run(**validated.model_dump())
        return result if isinstance(result, str) else str(result)

    def to_openai_tool(self) -> dict[str, Any]:
        """OpenAI-style function definition for this tool."""
        schema = self.args_schema.model_json_schema()
        properties: dict[str, Any] = {}
        for prop_name, prop in (schema.get("properties") or {}).items():
            properties[prop_name] = {k: v for k, v in prop.items() if k != "title"}
        required = [n for n in schema.get("required", []) if n in properties]
        parameters: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            parameters["required"] = required
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description[:1024],
                "parameters": parameters,
            },
        }
