from collections import Counter
from typing import Dict, List

import numpy as np

from src.lexical.indexer import get_document_frequency


class TfIdf:
    def __init__(
        self,
        index: Dict,
        do_tf: bool = False,
        do_idf: bool = False,
        field: str = "document",
    ):
        self.index = index
        self.do_tf = do_tf
        self.do_idf = do_idf
        self.field = field

        assert do_tf or do_idf
        assert field in ["document", "title", "abstract"]
        self.total_docs = index["corpus"]["unique_documents"]

    def __call__(self, query: List[int], text: List[int]) -> float:
        token2tf = Counter(text)
        score = 0

        for token in query:
            if token in token2tf:
                tf = np.log(1 + token2tf[token]) if self.do_tf else 1.0
                idf = self.smooth_idf(token) if self.do_idf else 1.0
                score += tf * idf

        return score

    def smooth_idf(self, token: int) -> float:
        docs = get_document_frequency(self.index, token, self.field, unique=True)
        return np.log((1 + self.total_docs) / (1 + docs))
