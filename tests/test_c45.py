import pytest
import pandas as pd
import pathlib

import numpy as np
from c45 import C45Tree

FIXTURE_DIR = pathlib.Path(__file__).parent.resolve() / "deepset_prompt_injections"


def test_c45tree():
    data_frame = pd.read_parquet(FIXTURE_DIR / "train-00000-of-00001-9564e8b05b4757ab.parquet")
    content = data_frame["text"].tolist()
    labels = data_frame["label"].tolist()
    labeled_documents = list(zip(content, labels))
    print("training")
    tree = C45Tree(labeled_documents)

    print("testing")
    test_data_frame = pd.read_parquet(FIXTURE_DIR / "test-00000-of-00001-701d16158af87368.parquet")
    test_content = test_data_frame["text"].tolist()
    test_labels = test_data_frame["label"].tolist()

    for content, label in zip(test_content, test_labels):
        assert tree.predict(content) == label, content
