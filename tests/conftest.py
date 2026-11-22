import sys

import pytest

from supportlens.train import main as train_main


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    """A tiny trained model (400 synthetic tickets, 1 epoch) shared by the tests that need artifacts."""
    out = tmp_path_factory.mktemp("art")
    old = sys.argv
    sys.argv = ["train", "--n", "400", "--epochs", "1", "--out", str(out)]
    try:
        train_main()
    finally:
        sys.argv = old
    return str(out)
