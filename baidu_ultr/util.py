from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import torch
import wget


class DatasetWriter:
    def __init__(
        self,
        half_precision: bool = True,
    ):
        self.half_precision = half_precision
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
            self.features[k] = list(torch.concat(self.features[k]).numpy())

        self.features["query_document_embedding"] = list(
            torch.vstack(self.embeddings).numpy()
        )

        df = pd.DataFrame(self.features)
        df.to_feather(path)


def download_model(directory: Path, name: str):
    path = directory / f"{name}.zip"
    url = f"https://huggingface.co/lixsh6/wsdm23_unbiased/resolve/main/pretrain/{name}.zip"
    wget.download(url, out=str(path))

    with ZipFile(path, "r") as f:
        f.extractall(directory)
