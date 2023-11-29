from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from baidu_ultr.data import BaiduTestDataset, BaiduTrainDataset
from baidu_ultr.util import DatasetWriter


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(config):
    print(OmegaConf.to_yaml(config))

    device = torch.device("cuda:0")
    data_directory = Path(config.data_directory)
    out_directory = Path(config.out_directory)

    if config.data_type == "train":
        in_path = data_directory / f"part-{config.train_part:05d}.gz"
        assert in_path.exists(), f"Train dataset not found at: {in_path}"

        out_file = f"part-{config.train_part}_split-{config.train_split_id}.feather"
        dataset = BaiduTrainDataset(
            in_path,
            config.train_split_id,
            config.train_queries_per_split,
            config.max_sequence_length,
            config.tokens.special_tokens,
            config.tokens.segment_types,
        )
    elif config.data_type == "val":
        in_path = data_directory / "annotation_data_0522.txt"
        out_file = f"validation.feather"

        dataset = BaiduTestDataset(
            in_path,
            config.max_sequence_length,
            config.tokens.special_tokens,
            config.tokens.segment_types,
        )
    else:
        raise ValueError("config.in_type must be in ['train', 'val']")

    dataset_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        pin_memory=True,
    )

    print(in_path)
    print(out_file)

    model = instantiate(config.model)
    model.load(device)
    writer = DatasetWriter(half_precision=config.half_precision)

    for i, batch in tqdm(enumerate(dataset_loader)):
        features, tokens, token_types = batch

        query_document_embedding = model(
            tokens,
            token_types,
        )

        assert not query_document_embedding.isnan().any(), "FOUND NAN"

        writer.add(features, query_document_embedding)

    writer.save(out_directory / out_file)


if __name__ == "__main__":
    main()
