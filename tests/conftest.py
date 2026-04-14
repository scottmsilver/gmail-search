import pytest

from gmail_search.config import load_config


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture
def test_config(tmp_path, data_dir):
    return load_config(config_path=tmp_path / "nonexistent.yaml", data_dir=data_dir)
