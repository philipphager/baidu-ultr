import gzip
from pathlib import Path

import torch
from torch.utils.data import IterableDataset

SPECIAL_TOKENS = {
    "PAD": 0,
    "SEP": 1,
    "CLS": 2,
    "MASK": 3,
}

TOKEN_TYPES = {
    "QUERY": 0,
    "TEXT": 1,
    "PAD": 1,  # See source code:
}


def preprocess(
        query: str,
        title: str,
        abstract: str,
        max_tokens: int,
):
    """
    Format BERT model input as:
    [CLS] + query + [SEP] + title + [SEP] + content + [SEP] + [PAD]
    """
    query_idx = split_idx(query)
    title_idx = split_idx(title)
    abstract_idx = split_idx(abstract)

    query_tokens = [SPECIAL_TOKENS["CLS"]] + query_idx + [SPECIAL_TOKENS["SEP"]]
    query_token_types = [TOKEN_TYPES["QUERY"]] * len(query_tokens)

    text_tokens = title_idx + [SPECIAL_TOKENS["SEP"]] + abstract_idx
    text_tokens += abstract_idx + [SPECIAL_TOKENS["SEP"]]
    text_token_types = [TOKEN_TYPES["TEXT"]] * len(text_tokens)

    tokens = query_tokens + text_tokens
    token_types = query_token_types + text_token_types

    padding = max(max_tokens - len(tokens), 0)
    tokens = tokens[:max_tokens] + padding * [SPECIAL_TOKENS["PAD"]]
    token_types = token_types[:max_tokens] + padding * [TOKEN_TYPES["PAD"]]

    tokens = torch.tensor(tokens, dtype=torch.int)
    attention_mask = tokens > SPECIAL_TOKENS["PAD"]
    token_types = torch.tensor(token_types, dtype=torch.int)

    return tokens, attention_mask, token_types


def split_idx(text, offset: int = 10):
    """
    Split tokens in Baidu dataset and convert to integer token ids.
    """
    return [int(t) + offset for t in text.split(b"\x01") if len(t.strip()) > 0]


class BaiduTrainDataset(IterableDataset):
    def __init__(
            self,
            path: Path,
            max_sequence_length: int,
    ):
        self.path = path
        self.max_sequence_length = max_sequence_length

    def __iter__(self):
        query_id = -1
        query = None

        with gzip.open(self.path, "rb") as f:
            for i, line in enumerate(f):
                columns = line.strip(b"\n").split(b"\t")
                is_query = len(columns) <= 3

                if is_query:
                    # Create surrogate query_id to reduce memory:
                    query_id += 1
                    query = columns[1]
                else:
                    title = columns[2]
                    abstract = columns[3]
                    click = int(columns[5])

                    tokens, attention_mask, token_types = preprocess(
                        query,
                        title,
                        abstract,
                        self.max_sequence_length,
                    )

                    yield query_id, click, tokens, attention_mask, token_types


class BaiduTestDataset(IterableDataset):
    def __init__(
            self,
            path: Path,
            max_sequence_length: int,
    ):
        self.path = path
        self.max_sequence_length = max_sequence_length

    def __iter__(self):
        query_id = -1
        current_query_id = None

        with open(self.path, "rb") as f:
            for i, line in enumerate(f):
                columns = line.strip(b"\n").split(b"\t")

                qid, query, title, abstract, label, frequency = columns

                if qid != current_query_id:
                    query_id += 1
                    current_query_id = qid

                tokens, attention_mask, token_types = preprocess(
                    query,
                    title,
                    abstract,
                    self.max_sequence_length,
                )

                yield query_id, int(label), tokens, attention_mask, token_types
