import gzip
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import IterableDataset

from src.const import TrainColumns, QueryColumns, TOKEN_OFFSET
from src.hash import md5


def preprocess(
    query: str,
    title: str,
    abstract: str,
    max_tokens: int,
    special_token: Dict[str, int],
    segment_type: Dict[str, int],
):
    """
    Format BERT model input as:
    [CLS] + query + [SEP] + title + [SEP] + content + [SEP] + [PAD]
    """
    query_idx = split_idx(query, TOKEN_OFFSET)
    title_idx = split_idx(title, TOKEN_OFFSET)
    abstract_idx = split_idx(abstract, TOKEN_OFFSET)

    query_tokens = [special_token["CLS"]] + query_idx + [special_token["SEP"]]
    query_token_types = [segment_type["QUERY"]] * len(query_tokens)

    text_tokens = title_idx + [special_token["SEP"]]
    text_tokens += abstract_idx + [special_token["SEP"]]
    text_token_types = [segment_type["TEXT"]] * len(text_tokens)

    tokens = query_tokens + text_tokens
    token_types = query_token_types + text_token_types

    padding = max(max_tokens - len(tokens), 0)
    tokens = tokens[:max_tokens] + padding * [special_token["PAD"]]
    token_types = token_types[:max_tokens] + padding * [segment_type["PAD"]]

    tokens = torch.tensor(tokens, dtype=torch.int)
    token_types = torch.tensor(token_types, dtype=torch.int)

    return tokens, token_types


def split_idx(text, offset: int):
    """
    Split tokens in Baidu dataset and convert to integer token ids.
    """
    return [int(t) + offset for t in text.split(b"\x01") if len(t.strip()) > 0]


class BaiduTrainDataset(IterableDataset):
    def __init__(
        self,
        path: Path,
        split_id: int,
        queries_per_split: int,
        max_sequence_length: int,
        special_token: Dict[str, int],
        segment_type: Dict[str, int],
        ignored_titles: List[bytes],
    ):
        self.path = path
        self.max_sequence_length = max_sequence_length
        self.begin_query = split_id * queries_per_split
        self.end_query = (split_id + 1) * queries_per_split
        self.special_token = special_token
        self.segment_type = segment_type
        self.ignored_titles = set(ignored_titles)
        print(f"Split:{split_id}, query_ids: [{self.begin_query}, {self.end_query})")

    def __iter__(self):
        query_no = -1
        query_id = None
        query = None
        skipped_docs = 0

        with gzip.open(self.path, "rb") as f:
            for i, line in enumerate(f):
                columns = line.strip(b"\n").split(b"\t")
                is_query = len(columns) <= 3

                if is_query:
                    query_no += 1
                    query_id = columns[QueryColumns.QID]
                    query = columns[QueryColumns.QUERY]
                else:
                    # Iterate over dataset until assigned query range is reached:
                    query_in_split = self.begin_query <= query_no < self.end_query

                    if query_in_split:
                        title = columns[TrainColumns.TITLE]

                        if title in self.ignored_titles:
                            skipped_docs += 1
                            continue

                        abstract = columns[TrainColumns.ABSTRACT]
                        url = columns[TrainColumns.URL_MD5]
                        position = int(columns[TrainColumns.POS])
                        media_type = columns[TrainColumns.MULTIMEDIA_TYPE]
                        media_type = int(media_type) if media_type != b"-" else 0
                        displayed_time = float(columns[TrainColumns.DISPLAYED_TIME])
                        slipoff = int(columns[TrainColumns.SLIPOFF_COUNT_AFTER_CLICK])
                        serp_height = int(columns[TrainColumns.SERP_HEIGHT])
                        click = int(columns[TrainColumns.CLICK])

                        features = {
                            "query_id": query_id.decode("utf-8"),
                            "query_md5": md5(query),
                            "url_md5": url.decode("utf-8"),
                            "text_md5": md5(title + b"\x01" + abstract),
                            "position": position,
                            "media_type": media_type,
                            "displayed_time": displayed_time,
                            "serp_height": serp_height,
                            "slipoff_count_after_click": slipoff,
                            "click": click,
                        }

                        tokens, token_types = preprocess(
                            query=query,
                            title=title,
                            abstract=abstract,
                            max_tokens=self.max_sequence_length,
                            special_token=self.special_token,
                            segment_type=self.segment_type,
                        )

                        yield features, tokens, token_types
                    elif query_no >= self.end_query:
                        # End of selected split reached, stop iterating
                        print("Split complete:", {"skipped_docs": skipped_docs})
                        break


class BaiduTestDataset(IterableDataset):
    def __init__(
        self,
        path: Path,
        max_sequence_length: int,
        special_token: Dict[str, int],
        segment_type: Dict[str, int],
        ignored_titles: List[bytes],
    ):
        self.path = path
        self.max_sequence_length = max_sequence_length
        self.special_token = special_token
        self.segment_type = segment_type
        self.ignored_titles = set(ignored_titles)

    def __iter__(self):
        skipped_docs = 0

        with open(self.path, "rb") as f:
            for i, line in enumerate(f):
                columns = line.strip(b"\n").split(b"\t")

                query_id, query, title, abstract, label, frequency_bucket = columns

                if title in self.ignored_titles:
                    skipped_docs += 1
                    continue

                features = {
                    "query_id": query_id.decode("utf-8"),
                    "query_md5": md5(query),
                    "text_md5": md5(title + b"\x01" + abstract),
                    "label": int(label),
                    "frequency_bucket": int(frequency_bucket),
                }

                tokens, token_types = preprocess(
                    query=query,
                    title=title,
                    abstract=abstract,
                    max_tokens=self.max_sequence_length,
                    special_token=self.special_token,
                    segment_type=self.segment_type,
                )

                yield features, tokens, token_types

        print("Split complete:", {"skipped_docs": skipped_docs})