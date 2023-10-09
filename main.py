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
    device = torch.device("mps")
    model_path = Path(config.model_directory) / config.model

    in_path = Path(config.in_directory) / config.in_file
    out_file = Path(config.in_file).with_suffix(".safetensors")
    out_path = Path(config.out_directory) / out_file

    writer = DatasetWriter(half_precision=config.half_precision)

    print(f"Loading: {in_path}")
    print(f"Output: {out_path}")

    if not model_path.exists():
        download_model(config.model_directory, config.model)

    if config.in_type == "train":
        dataset = BaiduTrainDataset(in_path, config.max_sequence_length)
    elif config.in_type == "test":
        dataset = BaiduTestDataset(in_path, config.max_sequence_length)
    else:
        raise ValueError("config.in_type must be in ['train', 'test']")

    dataset_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        pin_memory=True,
    )

    model = BertModel.from_pretrained(model_path, local_files_only=True)
    model.to(device)
    torch.compile(model)

    for i, batch in tqdm(enumerate(dataset_loader)):
        query_ids, clicks, tokens, attention_mask, token_types = batch

        model_output = model(
            tokens.to(device),
            attention_mask.to(device),
            token_types.to(device),
        )
        features = model_output.pooler_output
        writer.add(query_ids, clicks, features)

    writer.save(out_path)


if __name__ == "__main__":
    main()
