from collections import Counter
from typing import Dict, List, Optional

import numpy as np


class QueryLikelihood:
    def __init__(
        self,
        index: Dict,
        smoothing: Optional[str] = None,
        alpha: float = 0.1,
        field: str = "document",
    ):
        self.index = index
        self.smoothing = smoothing
        self.alpha = alpha
        self.field = field

        assert smoothing in ["jelinek-mercer", "dirichlet", None]
        assert field in ["document", "title", "abstract"]
        self.total_tokens = index["corpus"][f"total_{field}_length"]

    def __call__(self, query: List[int], text: List[int]):
        token2tf = Counter(text)
        doc_length = len(text)
        score = 0

        if self.smoothing == "jelinek-mercer":
            alpha = self.alpha
        elif self.smoothing == "dirichlet":
            alpha = self.alpha / (doc_length + self.alpha)
        else:
            alpha = 0

        for token in query:
            tf = token2tf.get(token, 0)
            df = self.index["tokens"][str(token)][self.field]["total_occurrences"]

            p_doc = tf / doc_length
            p_corpus = df / self.total_tokens
            score += np.log(alpha * p_corpus + (1 - alpha) * p_doc)

        return score
