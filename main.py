from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel

from baidu_ultr.data import BaiduDataset
from baidu_ultr.util import write_svmlight_file, download_model

if __name__ == "__main__":
    in_file = Path("data") / "part-00001.gz"
    out_file = Path("output") / "part-00001.txt"

    model = "base_group2"
    model_directory = Path("models")

    device = torch.device("cuda:0")
    batch_size = 64
    max_sequence_length = 128
    model_path = model_directory / model

    if not model_path.exists():
        download_model(model_directory, model)

    train_dataset = BaiduDataset(in_file, max_sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1)

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

        with open(out_file, "ab") as file:
            write_svmlight_file(query_ids, features, clicks, file)
