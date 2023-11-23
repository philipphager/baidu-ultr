from pathlib import Path

import hydra
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from baidu_ultr.const import SEGMENT_TYPES, BAIDU_SPECIAL_TOKENS
from baidu_ultr.data import BaiduTestDataset, BaiduTrainDataset
from baidu_ultr.model.tencent import TencentModel
from baidu_ultr.util import download_model, DatasetWriter


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(config):
    device = torch.device("cuda:0")
    model_path = Path(config.model_directory) / config.model
    data_directory = Path(config.data_directory)
    out_directory = Path(config.out_directory)

    if not model_path.exists():
        download_model(config.model_directory, config.model)

    if config.data_type == "train":
        in_path = data_directory / f"part-{config.train_part:05d}.gz"
        assert in_path.exists(), f"Train dataset not found at: {in_path}"

        out_file = f"part-{config.train_part}_split-{config.train_split_id}.feather"
        dataset = BaiduTrainDataset(
            in_path,
            config.train_split_id,
            config.train_queries_per_split,
            config.max_sequence_length,
            BAIDU_SPECIAL_TOKENS,
            SEGMENT_TYPES,
        )
    elif config.data_type == "val":
        in_path = data_directory / "annotation_data_0522.txt"
        out_file = f"validation.feather"

        dataset = BaiduTestDataset(in_path, config.max_sequence_length)
    else:
        raise ValueError("config.in_type must be in ['train', 'val']")

    dataset_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        pin_memory=True,
    )

    print(in_path)
    print(out_file)

    model = TencentModel(model_path, device)
    model.load()

    writer = DatasetWriter(half_precision=config.half_precision)

    for i, batch in tqdm(enumerate(dataset_loader)):
        features, tokens, token_types = batch

        query_document_embedding = model(
            tokens,
            token_types,
        )
        writer.add(features, query_document_embedding)

    writer.save(out_directory / out_file)


if __name__ == "__main__":
    main()
