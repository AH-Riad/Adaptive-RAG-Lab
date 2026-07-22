from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Document:
    """
    Represents an original document before chunking.
    """

    id: str
    source: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)