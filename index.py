from pathlib import Path

import hydra
from omegaconf import OmegaConf

from baidu_ultr.const import MISSING_TITLE, WHAT_OTHER_PEOPLE_SEARCHED_TITLE
from baidu_ultr.model.ltr.indexer import Indexer


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(config):
    print(OmegaConf.to_yaml(config))

    directory = Path(config.data_directory)
    train_files = [f for f in directory.glob("part-*")]
    output_path = Path(config.out_directory) / "index.json"

    indexer = Indexer(
        train_files,
        output_path,
        ignored_titles=[MISSING_TITLE, WHAT_OTHER_PEOPLE_SEARCHED_TITLE],
    )
    indexer()


if __name__ == '__main__':
    main()
