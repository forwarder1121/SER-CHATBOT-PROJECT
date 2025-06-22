from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Document:
    """문서 데이터를 표현하는 클래스"""
    page_content: str
    metadata: Dict[str, Any]

    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {} 