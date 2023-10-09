from pathlib import Path
from zipfile import ZipFile

import torch
import wget
from safetensors.torch import save_file


class DatasetWriter:
    def __init__(
            self,
            half_precision: bool = True,
    ):
        self.half_precision = half_precision
        self.query_ids = []
        self.labels = []
        self.features = []

    def add(self, query_ids, labels, features):
        if self.half_precision:
            # Encode datatypes to reduce memory
            query_ids = query_ids.to(torch.int)
            labels = labels.to(torch.short)
            features = features.to(torch.float16)

        # Ensure tensors are moved out of GPU memory
        self.query_ids.append(query_ids.cpu().detach())
        self.labels.append(labels.cpu().detach())
        self.features.append(features.cpu().detach())

    def save(self, path: Path):
        self.query_ids = torch.concat(self.query_ids)
        self.labels = torch.concat(self.labels)
        self.features = torch.vstack(self.features)

        save_file(
            {
                "query_ids": self.query_ids,
                "labels": self.labels,
                "features": self.features,
            },
            path,
        )


def download_model(directory: Path, name: str):
    path = directory / f"{name}.zip"
    url = f"https://huggingface.co/lixsh6/wsdm23_unbiased/resolve/main/pretrain/{name}.zip"
    wget.download(url, out=str(path))

    with ZipFile(path, "r") as f:
        f.extractall(directory)
