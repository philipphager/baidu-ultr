from pathlib import Path

import hydra
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel

from baidu_ultr.data import BaiduTestDataset, BaiduTrainDataset
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

        out_file = f"part-{config.train_part}_split-{config.train_split_id}.safetensors"
        dataset = BaiduTrainDataset(
            in_path,
            config.train_split_id,
            config.train_queries_per_split,
            config.max_sequence_length,
        )
    elif config.data_type == "val":
        in_path = data_directory / "annotation_data_0522.txt"
        out_file = f"validation.safetensors"

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

    model = BertModel.from_pretrained(model_path, local_files_only=True)
    model.to(device)
    torch.compile(model)

    writer = DatasetWriter(half_precision=config.half_precision)

    for i, batch in tqdm(enumerate(dataset_loader)):
        query_ids, clicks, tokens, attention_mask, token_types = batch

        model_output = model(
            tokens.to(device),
            attention_mask.to(device),
            token_types.to(device),
        )
        features = model_output.pooler_output
        writer.add(query_ids, clicks, features)

    writer.save(out_directory / out_file)


if __name__ == "__main__":
    main()
