from typing import Type, Dict, Any
from .component import Component

class ComponentRegistry:
    """
    A registry to dynamically store and instantiate components.
    This prevents hardcoding classes into your execution scripts.
    """
    def __init__(self):
        self._registry: Dict[str, Type[Component]] = {}

    def register(self, name: str, component_class: Type[Component]) -> None:
        """Registers a component class under a specific string name."""
        if not issubclass(component_class, Component):
            raise TypeError(f"Cannot register '{name}': Must be a subclass of Component.")
        self._registry[name] = component_class

    def get(self, name: str, **kwargs: Any) -> Component:
        """Instantiates and returns a component by its registered name."""
        if name not in self._registry:
            raise KeyError(f"Component '{name}' is not registered.")
        
        component_class = self._registry[name]
        return component_class(**kwargs)

registry = ComponentRegistry()