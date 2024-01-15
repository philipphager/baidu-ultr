from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import BaiduTrainDataset, BaiduTestDataset
from src.model.baidu import BaiduModel
from src.model.tencent import TencentModel
from test import tencent_dataloader, baidu_dataloader
from test.baidu_dataloader import TrainDataset

# Replace with path to Baidu dataset:
DATA_DIRECTORY = Path("data/")

TENCENT_SPECIAL_TOKENS = {
    "PAD": 0,
    "SEP": 1,
    "CLS": 2,
    "MASK": 3,
}

BAIDU_SPECIAL_TOKENS = {
    "CLS": 0,
    "SEP": 1,
    "PAD": 2,
    "MASK": 3,
}

SEGMENT_TYPES = {
    "QUERY": 0,
    "TEXT": 1,
    "PAD": 1,  # See source code:
}


def test_baidu_click_dataset():
    dataset = BaiduTrainDataset(
        DATA_DIRECTORY / "part-00001.gz",
        split_id=0,
        queries_per_split=10,
        max_sequence_length=128,
        special_token=BAIDU_SPECIAL_TOKENS,
        segment_type=SEGMENT_TYPES,
        drop_missing_docs=False,
        skip_what_others_searched=False,
    )
    loader = DataLoader(dataset, batch_size=100)
    actual_batch = next(iter(loader))
    features, tokens, token_types = actual_batch
    mask = BaiduModel.mask_padding(tokens, BAIDU_SPECIAL_TOKENS)

    original_dataset = TrainDataset(DATA_DIRECTORY, buffer_size=100, max_seq_len=128)
    original_loader = DataLoader(original_dataset, batch_size=100)
    expected_batch = next(iter(original_loader))
    (
        expected_tokens,
        expected_token_types,
        expected_mask,
        expected_clicks,
    ) = expected_batch

    assert torch.eq(tokens, expected_tokens).all()
    assert torch.eq(token_types, expected_token_types).all()
    assert torch.eq(mask, expected_mask).all()
    assert torch.eq(features["click"], expected_clicks).all()


def test_baidu_annotation_dataset():
    dataset = BaiduTestDataset(
        DATA_DIRECTORY / "annotation_data_0522.txt",
        max_sequence_length=128,
        special_token=BAIDU_SPECIAL_TOKENS,
        segment_type=SEGMENT_TYPES,
        drop_missing_docs=False,
        skip_what_others_searched=False,
    )
    loader = DataLoader(dataset, batch_size=100)
    actual_batch = next(iter(loader))
    features, tokens, token_types = actual_batch
    mask = BaiduModel.mask_padding(tokens, BAIDU_SPECIAL_TOKENS)

    original_dataset = baidu_dataloader.TestDataset(
        DATA_DIRECTORY / "annotation_data_0522.txt",
        max_seq_len=128,
        data_type="annotate",
        buffer_size=100,
    )
    original_loader = DataLoader(original_dataset, batch_size=100)
    expected_tokens, expected_token_types, expected_mask = next(iter(original_loader))
    expected_labels = torch.tensor(original_dataset.total_labels[:100])

    assert torch.eq(tokens, expected_tokens).all()
    assert torch.eq(token_types, expected_token_types).all()
    assert torch.eq(mask, expected_mask).all()
    assert torch.eq(features["label"], expected_labels).all()


def test_tencent_click_dataset():
    dataset = BaiduTrainDataset(
        DATA_DIRECTORY / "part-00001.gz",
        split_id=0,
        queries_per_split=10,
        max_sequence_length=128,
        special_token=TENCENT_SPECIAL_TOKENS,
        segment_type=SEGMENT_TYPES,
        drop_missing_docs=False,
        skip_what_others_searched=False,
    )

    loader = DataLoader(dataset, batch_size=100)
    actual_batch = next(iter(loader))
    features, tokens, token_types = actual_batch
    mask = TencentModel.mask_attention(tokens, TENCENT_SPECIAL_TOKENS)

    original_dataset = tencent_dataloader.TestDataset(
        DATA_DIRECTORY / "part-00001.gz",
        max_seq_len=128,
        data_type="click",
        buffer_size=100,
    )

    for i in range(100):
        expected_row = original_dataset[i]

        expected_tokens = torch.tensor(expected_row["src_input"])
        expected_token_types = torch.tensor(expected_row["segment"])
        expected_clicks = torch.tensor(expected_row["label"])
        # See: https://github.com/lixsh6/Tencent_wsdm_cup2023/blob/270ff4afdc6492b223e65c16de996e139ff7cf21/pytorch_unbias/pretrain/dataset.py#L259C35-L259C58
        expected_mask = expected_tokens > 0

        assert torch.eq(tokens[i], expected_tokens).all()
        assert torch.eq(token_types[i], expected_token_types).all()
        assert torch.eq(mask[i], expected_mask).all()
        assert torch.eq(features["click"][i], expected_clicks).all()


def test_tencent_annotation_dataset():
    dataset = BaiduTestDataset(
        DATA_DIRECTORY / "annotation_data_0522.txt",
        max_sequence_length=128,
        special_token=TENCENT_SPECIAL_TOKENS,
        segment_type=SEGMENT_TYPES,
        drop_missing_docs=False,
        skip_what_others_searched=False,
    )

    loader = DataLoader(dataset, batch_size=100)
    actual_batch = next(iter(loader))
    features, tokens, token_types = actual_batch
    mask = TencentModel.mask_attention(tokens, TENCENT_SPECIAL_TOKENS)

    original_dataset = tencent_dataloader.TestDataset(
        DATA_DIRECTORY / "annotation_data_0522.txt",
        max_seq_len=128,
        data_type="annotate",
        buffer_size=100,
    )

    for i in range(100):
        expected_row = original_dataset[i]

        expected_tokens = torch.tensor(expected_row["src_input"])
        expected_token_types = torch.tensor(expected_row["segment"])
        expected_label = torch.tensor(expected_row["label"])
        # See: https://github.com/lixsh6/Tencent_wsdm_cup2023/blob/270ff4afdc6492b223e65c16de996e139ff7cf21/pytorch_unbias/pretrain/dataset.py#L259C35-L259C58
        expected_mask = expected_tokens > 0

        assert torch.eq(tokens[i], expected_tokens).all()
        assert torch.eq(token_types[i], expected_token_types).all()
        assert torch.eq(mask[i], expected_mask).all()
        assert torch.eq(features["label"][i], expected_label).all()
