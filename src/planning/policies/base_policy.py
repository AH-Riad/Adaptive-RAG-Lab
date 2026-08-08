from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.adaptive_context import AdaptiveContext
from src.planning.retrieval_plan import RetrievalPlan
from src.planning.policy_result import PolicyResult


class BasePolicy(ABC):
    """
    Base class for every planning policy.
    """

    @abstractmethod
    def apply(
        self,
        context: AdaptiveContext,
        plan: RetrievalPlan,
    ) -> RetrievalPlan:
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__ 
    
    @property
    def version(self) -> str:
        return "1.0"

    @property
    def description(self) -> str:
        return "Generic base policy."