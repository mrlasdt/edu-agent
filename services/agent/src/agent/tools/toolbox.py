"""
ToolBox - the tools one turn offers the model, from one or more tool sets.

The math tools and the notes tool are separate sets with their own contract and their own
client. A turn may offer either or both, so the loop is handed one list of specs and one
`call` that sends each request to the set that owns that name. A name nobody owns is an error
result that lists the valid names, so a model that invents a tool is told what exists.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

Handler = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


class ToolBox:
    def __init__(self, *sets: tuple[list[dict[str, Any]], Handler]) -> None:
        self.specs: list[dict[str, Any]] = []
        self._owner: dict[str, Handler] = {}
        for specs, handler in sets:
            for spec in specs:
                name = str(spec["function"]["name"])
                if name in self._owner:
                    raise ValueError(f"two tool sets both define {name!r}")
                self._owner[name] = handler
                self.specs.append(spec)

    @property
    def names(self) -> list[str]:
        return list(self._owner)

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handler = self._owner.get(name)
        if handler is None:
            return {"error": f"no tool named {name!r}. The tools are: {', '.join(self.names)}."}
        return await handler(name, arguments)
