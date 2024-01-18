from collections import Counter
from typing import Dict, List

import numpy as np


class BM25:
    def __init__(
        self,
        index: Dict,
        k1: float = 1.2,
        b: float = 0.75,
        field: str = "document",
    ):
        self.index = index
        self.k1 = k1
        self.b = b
        self.field = field

        assert field in ["document", "title", "abstract"]
        self.total_docs = index["corpus"]["unique_documents"]
        self.total_length = index["corpus"][f"total_{field}_length"]
        self.avg_length = self.total_length / self.total_docs

    def __call__(self, query: List[int], text: List[int]):
        score = 0
        token2tf = Counter(text)
        length_norm = 1 - self.b + self.b * len(text) / self.avg_length

        for token in query:
            tf = token2tf.get(token, 0)
            idf = self.idf(token)
            score += idf * (tf * (self.k1 + 1) / (tf + self.k1 * length_norm))

        return score

    def idf(self, token: int):
        docs = self.index["tokens"][str(token)][self.field]["unique_occurrences"]
        return np.log(1 + (self.total_docs - docs + 0.5) / (docs + 0.5))
