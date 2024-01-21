from collections import defaultdict
from pathlib import Path
from typing import List, Any
from zipfile import ZipFile

import pandas as pd
import torch
import wget
from pyarrow import Tensor

from src.const import TOKEN_OFFSET
from src.data import split_idx


class DatasetWriter:
    def __init__(
        self,
        half_precision: bool,
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
        df.to_feather(path)


def flatten(lists: List[List[Any]]) -> List[Any]:
    return [item for _list in lists for item in _list]


def download_model(directory: Path, name: str):
    path = directory / f"{name}.zip"
    url = f"https://huggingface.co/lixsh6/wsdm23_unbiased/resolve/main/pretrain/{name}.zip"
    wget.download(url, out=str(path))

    with ZipFile(path, "r") as f:
        f.extractall(directory)
