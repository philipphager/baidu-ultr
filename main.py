from pathlib import Path

import hydra
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel

from baidu_ultr.data import BaiduTestDataset, BaiduTrainDataset
from baidu_ultr.util import download_model


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(config):
    device = torch.device("mps")
    model_path = Path(config.model_directory) / config.model

    if not model_path.exists():
        download_model(config.model_directory, config.model)

    if config.feedback == "click":
        dataset = BaiduTrainDataset(config.in_path, config.max_sequence_length)
    elif config.feedback == "rating":
        dataset = BaiduTestDataset(config.in_path, config.max_sequence_length)
    else:
        raise ValueError("config.file_type must be train or validation")

    dataset_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        pin_memory=True,
    )

    model = BertModel.from_pretrained(model_path, local_files_only=True)
    model.to(device)
    torch.compile(model)

    Path(config.out_path).unlink(missing_ok=True)

    for i, batch in tqdm(enumerate(dataset_loader)):
        query_ids, clicks, tokens, attention_mask, token_types = batch

        model_output = model(
            tokens.to(device),
            attention_mask.to(device),
            token_types.to(device),
        )
        features = model_output.pooler_output


if __name__ == "__main__":
    main()
