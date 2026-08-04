from abc import ABC, abstractmethod
from typing import Any
from .adaptive_context import AdaptiveContext

class Component(ABC):
    """
    Base class for all pipeline modules.
    Whether it's a QueryAnalyzer, DenseRetriever, or Generator,
    they all share this exact same interface.
    """
    
    @abstractmethod
    def run(self, context: AdaptiveContext) -> AdaptiveContext:
        """
        Executes the component's logic, modifies the context in-place, 
        and returns it.
        """
        raise NotImplementedError("Every component must implement the 'run' method.")