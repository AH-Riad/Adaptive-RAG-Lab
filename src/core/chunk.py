from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Chunk:
    """
    Represents a chunk extracted from a document.
    """

    id: str
    document_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)