from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import torch
import wget


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
            if "md5" in k or k == "query_id":
                self.features[k] = [h for hashes in v for h in hashes]
            else:
                self.features[k] = list(torch.concat(self.features[k]).numpy())

        self.features["query_document_embedding"] = list(
            torch.vstack(self.embeddings).numpy()
        )
        df = pd.DataFrame(self.features)
        df = self.filter_queries(df)
        df.to_feather(path)

    def filter_queries(self, df):
        query_df = df.groupby("query_id").agg(n_docs=("text_md5", "count")).reset_index()

        n_queries_before = len(query_df)
        query_df = query_df[query_df.n_docs >= self.min_docs_per_query]
        n_queries_after = len(query_df)
        print(
            f"Dropped {n_queries_before - n_queries_after} queries",
            f"with less than {self.min_docs_per_query} docs",
        )

        return df[df.query_id.isin(set(query_df.query_id))]


def download_model(directory: Path, name: str):
    path = directory / f"{name}.zip"
    url = f"https://huggingface.co/lixsh6/wsdm23_unbiased/resolve/main/pretrain/{name}.zip"
    wget.download(url, out=str(path))

    with ZipFile(path, "r") as f:
        f.extractall(directory)
