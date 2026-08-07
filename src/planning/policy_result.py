from dataclasses import dataclass


@dataclass
class PolicyResult:
    """
    Represents the output of a single policy.
    """

    policy_name: str

    decision: str

    confidence: float

    reason: str