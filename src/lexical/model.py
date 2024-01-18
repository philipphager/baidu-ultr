from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from src.const import TOKEN_OFFSET
from src.data import split_idx
from src.lexical.bm25 import BM25
from src.lexical.indexer import load_index
from src.lexical.query_likelihood import QueryLikelihood
from src.lexical.tf_idf import TfIdf


class LexicalModel:
    def __init__(self, index_path: Path):
        index = load_index(index_path)
        self.bm25 = BM25(index, k1=1.2, b=0.75)
        self.title_bm25 = BM25(index, k1=1.2, b=0.75, field="title")
        self.abstract_bm25 = BM25(index, k1=1.2, b=0.75, field="abstract")
        self.tf_idf = TfIdf(index, do_tf=True, do_idf=True)
        self.tf = TfIdf(index, do_tf=True)
        self.idf = TfIdf(index, do_idf=True)
        self.ql = QueryLikelihood(index)
        self.ql_jm_short = QueryLikelihood(index, smoothing="jelinek-mercer", alpha=0.1)
        self.ql_jm_long = QueryLikelihood(index, smoothing="jelinek-mercer", alpha=0.7)
        self.ql_dirichlet = QueryLikelihood(index, smoothing="dirichlet", alpha=256)

    def __call__(self, batch) -> Dict[str, List]:
        results = defaultdict(lambda: [])
        n_batch = len(batch["query"])

        for i in range(n_batch):
            query = split_idx(batch["query"][i], TOKEN_OFFSET)
            title = split_idx(batch["title"][i], TOKEN_OFFSET)
            abstract = split_idx(batch["abstract"][i], TOKEN_OFFSET)
            document = title + abstract

            results["bm25"].append(self.bm25(query, document))
            results["title_bm25"].append(self.title_bm25(query, title))
            results["abstract_bm25"].append(self.abstract_bm25(query, abstract))
            results["tf_idf"].append(self.tf_idf(query, document))
            results["tf"].append(self.tf(query, document))
            results["idf"].append(self.idf(query, document))
            results["ql_jelinek_mercer_short"].append(self.ql_jm_short(query, document))
            results["ql_jelinek_mercer_long"].append(self.ql_jm_long(query, document))
            results["ql_dirichlet"].append(self.ql_dirichlet(query, document))
            results["query_length"].append(len(query))
            results["document_length"].append(len(document))
            results["title_length"].append(len(title))
            results["abstract_length"].append(len(abstract))

        return results
