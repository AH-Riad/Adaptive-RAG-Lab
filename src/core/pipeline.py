from __future__ import annotations

from src.core.adaptive_context import AdaptiveContext
from src.core.component import Component


class AdaptivePipeline:
    """
    Executes pipeline components sequentially.
    """

    def __init__(self):
        self.components: list[Component] = []

    def add(self, component: Component) -> None:
        self.components.append(component)

    def run(self, context: AdaptiveContext) -> AdaptiveContext:

        for component in self.components:

            context.add_log(
                f"Running {component.__class__.__name__}"
            )

            context = component.run(context)

        return context