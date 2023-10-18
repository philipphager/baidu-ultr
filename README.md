# Baidu Unbiased Learning to Rank - 606K Dataset
At NeurIPS 2022, [Baidu released the first large-scale click dataset](A Large Scale Search Dataset for Unbiased Learning to Rank
) for unbiased learing to rank. The full dataset contains over 1.2 B sessions of users browsing the Baidu search engine. The dataset comprises a.o., user clicks, skips, dwell-time, and the original query and document text. Traditionally, the unbiased learning to rank community uses query-document feature representations (e.g., [MSLR30K](https://www.microsoft.com/en-us/research/project/mslr/), [Istella-S](http://quickrank.isti.cnr.it/istella-dataset/), or [Yahoo! Webscope](https://webscope.sandbox.yahoo.com/catalog.php?datatype=c)), small neural network models, and focuses more on the aspect of removing click biases.

To make the massive Baidu dataset more accessible, we encode the query and document text into query-document embeddings using the winning BERT cross-encoder model from the WSDM Cup 2023. As BERT embeddings with 768 dimensions use a lot of memory, we encode them with half-precision floats and compress the dataset using [Arrow feather](https://arrow.apache.org/docs/python/feather.html). 

This dataset focuses only on the first partition (partition-0) from the [original dataset](https://drive.google.com/drive/folders/1Q3bzSgiGh1D5iunRky6mb89LpxfAO73J). It comprises 606k user sessions with clicks for training and the complete Baidu validation set containing expert annotations (the test set from the WSDM Cup 2023 was not released publicly).

## Usage
### I. Load training clicks
Load clicks from the training dataset (patition 0 / 1,999) of Baidu ULTR. The first partition contains 606k search queries. We converted the query and document text from the original dataset to query-document features using the winning [BERT cross-encoder model](https://github.com/lixsh6/Tencent_wsdm_cup2023/tree/main/pytorch_unbias/) from the WSDM Cup 2023.

```
from datasets import load_dataset
from torch.utils.data import DataLoader

dataset = load_dataset("philipphager/baidu-ultr-606k", name="clicks", split="train")
dataset.set_format("torch")
loader = DataLoader(dataset, collate_fn=collate_clicks, batch_size=8)
```

### II. Load expert annotations for validation
Only the validation set of the Baidu ULTR dataset is public. It also contains different columns from the training set, so you need to adjust your collate function accordingly:

```
from datasets import load_dataset
from torch.utils.data import DataLoader

val_dataset = load_dataset("philipphager/baidu-ultr-606k", name="annotations", split="validation")
val_dataset.set_format("torch")
loader = DataLoader(val_dataset, collate_fn=collate_annotations, batch_size=8)
```

### III. Batching queries
You can use the following `collate_fn` method to create a batch of queries (with differnet number of documents) and to select which columns to load from the training set.
```
from collections import defaultdict
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence

def collate_clicks(samples: List):
    """
    Pad a batch of queries to the size of the query with the most documents.
    """
    batch = defaultdict(lambda: [])

    for sample in samples:
        # Select information to load for each query:
        # Available are: ["query_id", "position", "click", "n", "query_document_embedding",
        # "media_type", "displayed_time", "serp_height", "slipoff_count_after_click"]
        batch["query_id"].append(sample["query_id"])
        batch["query_document_embedding"].append(sample["query_document_embedding"])
        batch["click"].append(sample["click"])
        batch["n"].append(sample["n"])

    # Convert to tensors and pad to document-level features:
    return {
        "query_id": torch.tensor(batch["query_id"]),
        "query_document_embedding": pad_sequence(
            batch["query_document_embedding"], batch_first=True
        ),
        "click": pad_sequence(batch["click"], batch_first=True),
        "n": torch.tensor(batch["n"]),
    }
```
