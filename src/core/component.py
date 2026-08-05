from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.adaptive_context import AdaptiveContext


class Component(ABC):
    """
    Base class for every pipeline component.
    """

    @abstractmethod
    def run(self, context: AdaptiveContext) -> AdaptiveContext:
        """
        Execute this component.

        Returns
        -------
        AdaptiveContext
            Updated context.
        """
        pass