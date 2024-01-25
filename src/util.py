from collections import defaultdict
from pathlib import Path
from typing import List, Any
from zipfile import ZipFile

import pandas as pd
import torch
import wget

from src.const import TOKEN_OFFSET
from src.data import split_idx


class DatasetWriter:
    def __init__(
        self,
        half_precision: bool,
        min_docs_per_query: int,
    ):
        self.half_precision = half_precision
        self.min_docs_per_query = min_docs_per_query
        self.features = defaultdict(lambda: [])
        self.embeddings = []

    def add(self, features, query_document_embedding):
        if self.half_precision:
            query_document_embedding = query_document_embedding.to(torch.float16)

        self.embeddings.append(query_document_embedding.cpu().detach())

        for k, v in features.items():
            self.features[k].append(v)

    def save(self, path: Path):
        for k, v in self.features.items():
            if k in [
                "query_no",
                "position",
                "media_type",
                "displayed_time",
                "serp_height",
                "slipoff_count_after_click",
                "click",
                "frequency_bucket",
                "label",
            ]:
                self.features[k] = list(torch.concat(self.features[k]).numpy())
            else:
                if k in ["query", "title", "abstract"]:
                    self.features[k] = [split_idx(t, TOKEN_OFFSET) for t in flatten(v)]
                else:
                    self.features[k] = flatten(v)

        self.features["query_document_embedding"] = list(
            torch.vstack(self.embeddings).numpy()
        )
        df = pd.DataFrame(self.features)
        df = self.filter_queries(df)
        df.to_feather(path)

    def filter_queries(self, df):
        n_queries_before = df["query_no"].nunique()
        df = df.groupby(["query_no"]).filter(
            lambda x: len(x) >= self.min_docs_per_query
        )
        n_queries_after = df["query_no"].nunique()
        print(
            f"Dropped {n_queries_before - n_queries_after} queries",
            f"with less than {self.min_docs_per_query} docs",
        )
        return df


def flatten(lists: List[List[Any]]) -> List[Any]:
    return [item for _list in lists for item in _list]


def download_model(directory: Path, name: str):
    path = directory / f"{name}.zip"
    url = f"https://huggingface.co/lixsh6/wsdm23_unbiased/resolve/main/pretrain/{name}.zip"
    wget.download(url, out=str(path))

    with ZipFile(path, "r") as f:
        f.extractall(directory)
