from pathlib import Path

import hydra
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(config):
    print(OmegaConf.to_yaml(config))

    ignored_columns = ["query_document_embedding"]

    data_directory = Path(config.data_directory)
    out_directory = Path(config.out_directory)

    parts = out_directory.glob("part-*.feather")
    parts = [p for p in parts if not p.name.startswith("part-0")]
    dfs = []

    for part in tqdm(parts):
        df = pd.read_feather(part)
        df = df.drop(columns=ignored_columns)
        dfs.append(df)

    df = pd.concat(dfs)
    df.to_feather(out_directory / "features.feather")


if __name__ == "__main__":
    main()
