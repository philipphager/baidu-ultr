from pathlib import Path
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.parquet as pq
import wget
from sklearn.datasets import dump_svmlight_file


class ParquetWriter:
    def __init__(self, path: Path):
        self.path = path
        self.schema = pa.schema(
            [
                ("query_id", pa.int32()),
                ("X", pa.list_(pa.float32())),
                ("y", pa.int8()),
            ]
        )
        self.writer = None

    def __enter__(self):
        self.writer = pq.ParquetWriter(
            self.path,
            self.schema,
        )

        return self

    def write(self, query_ids, features, label):
        batch = pa.record_batch(
            [
                list(query_ids.numpy()),
                list(features.cpu().detach().numpy()),
                list(label.numpy()),
            ],
            schema=self.schema,
        )

        self.writer.write_batch(batch)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()


class SvmLightWriter:
    def __init__(self, path: Path):
        self.path = path
        self.file = None

    def __enter__(self):
        self.file = open(self.path, "ab")
        return self

    def write(self, query_ids, features, label):
        dump_svmlight_file(
            X=features.cpu().detach().numpy(),
            y=label.numpy(),
            query_id=query_ids.numpy(),
            f=self.file,
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()


def download_model(directory: Path, name: str):
    path = directory / f"{name}.zip"
    url = f"https://huggingface.co/lixsh6/wsdm23_unbiased/resolve/main/pretrain/{name}.zip"
    wget.download(url, out=str(path))

    with ZipFile(path, "r") as f:
        f.extractall(directory)
