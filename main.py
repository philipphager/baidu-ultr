from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import BaiduTestDataset, BaiduTrainDataset
from src.lexical.model import LexicalModel
from src.util import DatasetWriter


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(config):
    print(OmegaConf.to_yaml(config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_directory = Path(config.data_directory)
    out_directory = Path(config.out_directory)
    out_directory.mkdir(parents=True, exist_ok=True)

    if config.data_type == "train":
        in_path = data_directory / f"part-{config.train_part:05d}.gz"
        out_file = f"part-{config.train_part}_split-{config.train_split_id}.feather"
        assert in_path.exists(), f"Train dataset not found at: {in_path}"

        dataset = BaiduTrainDataset(
            path=in_path,
            split_id=config.train_split_id,
            queries_per_split=config.train_queries_per_split,
            max_sequence_length=config.max_sequence_length,
            special_token=config.tokens.special_tokens,
            segment_type=config.tokens.segment_types,
            drop_missing_docs=config.drop_missing_docs,
            skip_what_others_searched=config.skip_what_others_searched,
        )
    elif config.data_type == "val":
        in_path = data_directory / "annotation_data_0522.txt"
        out_file = "validation.feather"
        assert in_path.exists(), f"Val dataset not found at: {in_path}"

        dataset = BaiduTestDataset(
            path=in_path,
            max_sequence_length=config.max_sequence_length,
            special_token=config.tokens.special_tokens,
            segment_type=config.tokens.segment_types,
            drop_missing_docs=config.drop_missing_docs,
            skip_what_others_searched=config.skip_what_others_searched,
        )
    else:
        raise ValueError("config.in_type must be in ['train', 'val']")

    dataset_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        pin_memory=True,
    )

    bert_model = instantiate(config.model, special_tokens=config.tokens.special_tokens)
    bert_model.load(device)
    lexical_model = LexicalModel(config.index_path)
    writer = DatasetWriter(
        half_precision=config.half_precision,
        min_docs_per_query=config.min_docs_per_query,
    )

    for i, batch in tqdm(enumerate(dataset_loader)):
        features, tokens, token_types = batch

        with torch.no_grad():
            query_document_embedding = bert_model(
                tokens,
                token_types,
            )

            # Add lexical ranking features, such as BM25, TF-IDF, or QL
            features = features | lexical_model(features)

        # assert not query_document_embedding.isnan().any()
        writer.add(features, query_document_embedding)

    writer.save(out_directory / out_file)


if __name__ == "__main__":
    main()
