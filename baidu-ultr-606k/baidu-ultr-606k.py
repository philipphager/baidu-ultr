from enum import Enum
from typing import List

import datasets
from pyarrow import feather

_CITATION = """\
@InProceedings{huggingface:dataset,
    title = {baidu-ultr-606k},
    author={Philipp Hager},
    year={2023}
}
"""

_DESCRIPTION = """\
Query-document vectors and clicks for the Baidu Unbiased Learning to Rank dataset used
at the WSDM23 cup. This dataset uses the winning BERT cross-encoder from Tencent
to compute query-document vectors (768 dims), mainly for ease of use and to enable
usage of simpler, smaller neural networks that are more common in ULTR research.

This dataset contains features for part-00000.gz of the Baidu dataset,
containing 589,824 queries and 6,271,536 documents. 
"""
_HOMEPAGE = "https://huggingface.co/datasets/philipphager/baidu-ultr-606k/"
_LICENSE = ""
_URL = "https://huggingface.co/datasets/philipphager/baidu-ultr-606k/"
_PARTS = 1
_SPLITS = 10


class Config(str, Enum):
    ANNOTATIONS = "annotations"
    CLICKS = "clicks"


class BaiduUltr606K(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.1.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="clicks",
            version=VERSION,
            description="Load clicks from the Baidu ULTR dataset",
        ),
        datasets.BuilderConfig(
            name="annotations",
            version=VERSION,
            description="Load expert annotations from the Baidu ULTR dataset",
        ),
    ]

    CLICK_FEATURES = datasets.Features(
        {
            "query_id": datasets.Value("int32"),
            "query_document_embedding": datasets.Array2D((None, 768), "float16"),
            "click": datasets.Sequence(datasets.Value("int32")),
            "n": datasets.Value("int32"),
            "position": datasets.Sequence(datasets.Value("int32")),
            "media_type": datasets.Sequence(datasets.Value("int32")),
            "displayed_time": datasets.Sequence(datasets.Value("float32")),
            "serp_height": datasets.Sequence(datasets.Value("int32")),
            "slipoff_count_after_click": datasets.Sequence(datasets.Value("int32")),
        }
    )

    ANNOTATION_FEATURES = datasets.Features(
        {
            "query_id": datasets.Value("int32"),
            "query_document_embedding": datasets.Array2D((None, 768), "float16"),
            "label": datasets.Sequence(datasets.Value("int32")),
            "n": datasets.Value("int32"),
            "frequency_bucket": datasets.Value("int32"),
        }
    )

    DEFAULT_CONFIG_NAME = Config.CLICKS

    def _info(self):
        if self.config.name == Config.CLICKS:
            features = self.CLICK_FEATURES
        elif self.config.name == Config.ANNOTATIONS:
            features = self.ANNOTATION_FEATURES
        else:
            raise ValueError("Config 'name' should be in ['clicks', 'annotations']")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        if self.config.name == Config.CLICKS:
            urls = [
                f"part-{p}_split-{s}.feather"
                for p in range(_PARTS)
                for s in range(_SPLITS)
            ]
            split = datasets.Split.TRAIN
            query_columns = [
                "query_id",
                "query_id",
            ]
            agg_columns = [
                "position",
                "click",
                "query_document_embedding",
                "media_type",
                "displayed_time",
                "serp_height",
                "slipoff_count_after_click",
            ]
        elif self.config.name == Config.ANNOTATIONS:
            urls = ["validation.feather"]
            split = datasets.Split.VALIDATION
            query_columns = [
                "query_id",
                "frequency_bucket",
            ]
            agg_columns = [
                "label",
                "query_document_embedding",
            ]
        else:
            raise ValueError("Config 'name' should be in ['clicks', 'annotations']")

        files = dl_manager.download(urls)

        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={
                    "files": files,
                    "query_columns": query_columns,
                    "agg_columns": agg_columns,
                },
            ),
        ]

    def _generate_examples(
        self,
        files: List[str],
        query_columns: List[str],
        agg_columns: List[str],
    ):
        """
        Reads dataset partitions and aggregates document features per query.
        :param files: List of .feather files to load from disk.
        :param query_columns: Columns with one value per query. E.g., query_id,
        frequency bucket, etc.
        :param agg_columns: Columns with one value per document that should be
        aggregated per query. E.g., click, position, query_document_embeddings, etc.
        :return:
        """
        for file in files:
            df = feather.read_feather(file)
            current_query_id = None
            sample_key = None
            sample = None

            for i in range(len(df)):
                row = df.iloc[i]

                if current_query_id != row["query_id"]:
                    if current_query_id is not None:
                        yield sample_key, sample

                    current_query_id = row["query_id"]
                    sample_key = f"{file}-{current_query_id}"
                    sample = {"n": 0}

                    for column in query_columns:
                        sample[column] = row[column]
                    for column in agg_columns:
                        sample[column] = []

                for column in agg_columns:
                    sample[column].append(row[column])

                sample["n"] += 1

            yield sample_key, sample