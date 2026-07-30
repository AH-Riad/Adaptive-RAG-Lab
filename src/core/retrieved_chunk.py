from dataclasses import dataclass
from typing import Dict

@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    metadata: Dict