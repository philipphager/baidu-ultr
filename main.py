from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.const import TITLES
from src.data import BaiduTestDataset, BaiduTrainDataset
from src.util import DatasetWriter


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(config):
    print(OmegaConf.to_yaml(config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_directory = Path(config.data_directory)
    out_directory = Path(config.out_directory)
    out_directory.mkdir(parents=True, exist_ok=True)

    # Optionally skip documents based on their title
    ignored_titles = [TITLES[t] for t in config.ignored_titles]

    if config.data_type == "train":
        in_path = data_directory / f"part-{config.train_part:05d}.gz"
        out_file = f"part-{config.train_part}_split-{config.train_split_id}.feather"
        assert in_path.exists(), f"Train dataset not found at: {in_path}"

        dataset = BaiduTrainDataset(
            in_path,
            config.train_split_id,
            config.train_queries_per_split,
            config.max_sequence_length,
            config.tokens.special_tokens,
            config.tokens.segment_types,
            ignored_titles,
        )
    elif config.data_type == "val":
        in_path = data_directory / "annotation_data_0522.txt"
        out_file = "validation.feather"
        assert in_path.exists(), f"Val dataset not found at: {in_path}"

        dataset = BaiduTestDataset(
            in_path,
            config.max_sequence_length,
            config.tokens.special_tokens,
            config.tokens.segment_types,
            ignored_titles,
        )
    else:
        raise ValueError("config.in_type must be in ['train', 'val']")

    dataset_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        pin_memory=True,
    )

    model = instantiate(config.model)
    model.load(device)
    writer = DatasetWriter(
        half_precision=config.half_precision,
        min_docs_per_query=config.min_docs_per_query,
    )

    for i, batch in tqdm(enumerate(dataset_loader)):
        features, tokens, token_types = batch

        with torch.no_grad():
            query_document_embedding = model(
                tokens,
                token_types,
            )

        assert not query_document_embedding.isnan().any()
        writer.add(features, query_document_embedding)

    writer.save(out_directory / out_file)


if __name__ == "__main__":
    main()
