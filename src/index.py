import gzip
import json
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict

from rbloom import Bloom
from tqdm import tqdm

from src.const import TrainColumns, TOKEN_OFFSET
from src.data import split_idx


class Indexer:
    def __init__(
            self,
            paths: List[Path],
            ignored_titles: List,
    ):
        self.paths = paths
        self.ignored_titles = set(ignored_titles)
        # Bloom filter to check if doc was already indexed before:
        self.indexed_urls = Bloom(1_000_000_000, 0.001)
        self.index = {
            "corpus": {
                "unique_documents": 0,
                "total_document_length": 0,
                "total_title_length": 0,
                "total_abstract_length": 0,
            },
            "tokens": defaultdict(
                lambda: {
                    "document": {"total_occurrences": 0, "unique_occurrences": 0},
                    "title": {"total_occurrences": 0, "unique_occurrences": 0},
                    "abstract": {"total_occurrences": 0, "unique_occurrences": 0},
                }
            ),
        }

    def __call__(self):
        for path in tqdm(self.paths, "Indexed files:"):
            with gzip.open(path, "rb") as f:
                for line in tqdm(f):
                    columns = line.strip(b"\n").split(b"\t")
                    is_query = len(columns) <= 3

                    if is_query:
                        # Only index documents
                        continue

                    title = columns[TrainColumns.TITLE]
                    abstract = columns[TrainColumns.ABSTRACT]
                    url = columns[TrainColumns.URL_MD5]

                    if title in self.ignored_titles or url in self.indexed_urls:
                        # Do not consider duplicate urls in index
                        continue

                    title = split_idx(title, TOKEN_OFFSET)
                    abstract = split_idx(abstract, TOKEN_OFFSET)
                    document = title + abstract

                    self.index["corpus"]["unique_documents"] += 1
                    self.index["corpus"]["total_document_length"] += len(document)
                    self.index["corpus"]["total_title_length"] += len(title)
                    self.index["corpus"]["total_abstract_length"] += len(abstract)

                    for token, freq in Counter(document).items():
                        stats = self.index["tokens"][token]["document"]
                        stats["unique_occurrences"] += 1
                        stats["total_occurrences"] += freq

                    for token, freq in Counter(title).items():
                        stats = self.index["tokens"][token]["title"]
                        stats["unique_occurrences"] += 1
                        stats["total_occurrences"] += freq

                    for token, freq in Counter(abstract).items():
                        stats = self.index["tokens"][token]["abstract"]
                        stats["unique_occurrences"] += 1
                        stats["total_occurrences"] += freq

                    self.indexed_urls.add(url)

        return self.index


def save_index(path: Path, index: Dict):
    with open(path, "w") as f:
        json.dump(index, f, indent=4, sort_keys=True)


def load_index(path: Path):
    with open(path, "r") as f:
        return json.load(f)
