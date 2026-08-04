from typing import List
from .component import Component
from .adaptive_context import AdaptiveContext

class Pipeline:
    """
    Chains components together and executes them sequentially.
    """
    def __init__(self):
        self.components: List[Component] = []

    def add(self, component: Component) -> "Pipeline":
        """
        Adds a component to the pipeline execution chain.
        Returns self to allow method chaining if desired.
        """
        if not isinstance(component, Component):
            raise TypeError("Only instances of 'Component' can be added to the Pipeline.")
        self.components.append(component)
        return self

    def run(self, context: AdaptiveContext) -> AdaptiveContext:
        """
        Passes the context sequentially through every registered component.
        """
        for component in self.components:
            # Each component modifies the context and passes it forward
            context = component.run(context)
        return context