from __future__ import annotations

from src.core.component import Component


class Registry:
    """
    Registers framework components.

    Example
    -------
    registry.register("dense", DenseRetriever)
    retriever = registry.create("dense")
    """

    def __init__(self):
        self._registry: dict[str, type[Component]] = {}

    def register(
        self,
        name: str,
        component: type[Component],
    ) -> None:

        self._registry[name] = component

    def create(self, name: str, *args, **kwargs) -> Component:

        if name not in self._registry:

            raise KeyError(
                f"'{name}' is not registered."
            )

        return self._registry[name](
            *args,
            **kwargs,
        )

    def list_components(self):

        return sorted(self._registry.keys())