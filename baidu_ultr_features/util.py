from pathlib import Path
from zipfile import ZipFile

import wget
from sklearn.datasets import dump_svmlight_file


def write_svmlight_file(
    query_ids,
    features,
    label,
    file,
):
    dump_svmlight_file(
        X=features.cpu().detach().numpy(),
        y=label.numpy(),
        query_id=query_ids.numpy(),
        f=file,
    )


def download_model(directory: Path, name: str):
    path = directory / f"{name}.zip"
    url = f"https://huggingface.co/lixsh6/wsdm23_unbiased/resolve/main/pretrain/{name}.zip"
    wget.download(url, out=path)

    with ZipFile(path, "r") as f:
        f.extractall(directory)
