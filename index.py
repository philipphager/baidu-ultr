from pathlib import Path

import hydra

from src.const import TITLES
from src.index import Indexer, save_index


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(config):
    directory = Path(config.data_directory)
    train_files = [f for f in sorted(directory.glob("part-*"))][:config.index_parts]
    output_path = Path(config.out_directory) / "index.json"

    # Optionally skip documents based on their title
    ignored_titles = [TITLES[t] for t in config.ignored_titles]

    indexer = Indexer(
        train_files,
        ignored_titles=ignored_titles,
    )
    index = indexer()
    save_index(output_path, index)


if __name__ == "__main__":
    main()
