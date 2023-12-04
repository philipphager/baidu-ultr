import gzip
from pathlib import Path
from typing import Dict

import hydra
import pandas as pd
from omegaconf import OmegaConf
from torch.utils.data import IterableDataset
from tqdm import tqdm

from baidu_ultr.const import TrainColumns, QueryColumns
from baidu_ultr.hash import md5


class TrainDatasetFeatures(IterableDataset):
    def __init__(
            self,
            path: Path,
            split_id: int,
            queries_per_split: int,
            max_sequence_length: int,
            special_token: Dict[str, int],
            segment_type: Dict[str, int],
    ):
        self.path = path
        self.max_sequence_length = max_sequence_length
        self.begin_query_id = split_id * queries_per_split
        self.end_query_id = (split_id + 1) * queries_per_split
        self.special_token = special_token
        self.segment_type = segment_type

        print(
            f"Split:{split_id}, query_ids: [{self.begin_query_id}, {self.end_query_id})"
        )

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
                    query = columns[QueryColumns.QUERY]
                else:
                    # Iterate over dataset until assigned query range is reached:
                    query_in_split = self.begin_query_id <= query_id < self.end_query_id

                    if query_in_split:
                        title = columns[TrainColumns.TITLE]
                        abstract = columns[TrainColumns.ABSTRACT]
                        url = columns[TrainColumns.URL_MD5]
                        position = int(columns[TrainColumns.POS])
                        media_type = columns[TrainColumns.MULTIMEDIA_TYPE]
                        media_type = int(media_type) if media_type != b"-" else 0
                        displayed_time = float(columns[TrainColumns.DISPLAYED_TIME])
                        slipoff = int(columns[TrainColumns.SLIPOFF_COUNT_AFTER_CLICK])
                        serp_height = int(columns[TrainColumns.SERP_HEIGHT])
                        click = int(columns[TrainColumns.CLICK])

                        yield {
                            "query_id": query_id,
                            "query_md5": md5(query),
                            "url_md5": url.decode("utf-8"),
                            "text_md5": md5(title + abstract),
                            "position": position,
                            "media_type": media_type,
                            "displayed_time": displayed_time,
                            "serp_height": serp_height,
                            "slipoff_count_after_click": slipoff,
                            "click": click,
                        }


class TestDatasetFeatures(IterableDataset):
    def __init__(
        self,
        path: Path,
        max_sequence_length: int,
        special_token: Dict[str, int],
        segment_type: Dict[str, int],
    ):
        self.path = path
        self.max_sequence_length = max_sequence_length
        self.special_token = special_token
        self.segment_type = segment_type

    def __iter__(self):
        query_id = -1
        current_query_id = None

        with open(self.path, "rb") as f:
            for i, line in enumerate(f):
                columns = line.strip(b"\n").split(b"\t")

                qid, query, title, abstract, label, frequency_bucket = columns

                if qid != current_query_id:
                    query_id += 1
                    current_query_id = qid

                yield {
                    "query_id": query_id,
                    "label": int(label),
                    "query_md5": md5(query),
                    "text_md5": md5(title + abstract),
                    "frequency_bucket": int(frequency_bucket),
                }


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(config):
    print(OmegaConf.to_yaml(config))

    data_directory = Path(config.data_directory)
    out_directory = Path(config.out_directory)

    in_path = data_directory / f"part-{config.train_part:05d}.gz"
    assert in_path.exists(), f"Train dataset not found at: {in_path}"

    if config.data_type == "train":

        out_file = f"features-part-{config.train_part}_split-{config.train_split_id}.csv"
        dataset = TrainDatasetFeatures(
            in_path,
            config.train_split_id,
            config.train_queries_per_split,
            config.max_sequence_length,
            config.tokens.special_tokens,
            config.tokens.segment_types,
        )
    elif config.data_type == "val":
        in_path = data_directory / "annotation_data_0522.txt"
        out_file = f"features-validation.csv"

        dataset = TestDatasetFeatures(
            in_path,
            config.max_sequence_length,
            config.tokens.special_tokens,
            config.tokens.segment_types,
        )
    else:
        raise ValueError("config.in_type must be in ['train', 'val']")

    rows = []

    for row in tqdm(dataset):
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_directory / out_file, index=False)


if __name__ == '__main__':
    main()
