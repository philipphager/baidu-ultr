from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel

from baidu_ultr.data import BaiduTestDataset
from baidu_ultr.util import download_model

if __name__ == "__main__":
    in_file = Path("data") / "annotation_data_0522.txt"
    out_file = Path("output") / "test.svm"

    model = "base_group2"
    model_directory = Path("models")

    device = torch.device("mps")
    batch_size = 8
    max_sequence_length = 128
    model_path = model_directory / model

    if not model_path.exists():
        download_model(model_directory, model)

    train_dataset = BaiduTestDataset(in_file, max_sequence_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
    )

    model = BertModel.from_pretrained(model_path, local_files_only=True)
    model.to(device)
    torch.compile(model)

    out_file.unlink(missing_ok=True)

    for i, batch in tqdm(enumerate(train_loader)):
        query_ids, clicks, tokens, attention_mask, token_types = batch

        model_output = model(
            tokens.to(device),
            attention_mask.to(device),
            token_types.to(device),
        )
        features = model_output.pooler_output
