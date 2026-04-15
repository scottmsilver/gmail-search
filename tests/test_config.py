from gmail_search.config import load_config


def test_load_config_defaults(tmp_path):
    """Config loads with all defaults when no file exists."""
    cfg = load_config(config_path=tmp_path / "nonexistent.yaml")
    assert cfg["budget"]["max_usd"] == 5.00
    assert cfg["embedding"]["model"] == "gemini-embedding-2-preview"
    assert cfg["embedding"]["dimensions"] == 3072
    assert cfg["attachments"]["max_file_size_mb"] == 10
    assert cfg["download"]["batch_size"] == 25
    assert cfg["search"]["default_top_k"] == 20
    assert cfg["server"]["port"] == 8080


def test_load_config_from_file(tmp_path):
    """Config loads from YAML and overrides defaults."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("budget:\n  max_usd: 20.00\n")
    cfg = load_config(config_path=config_file)
    assert cfg["budget"]["max_usd"] == 20.00
    # Other defaults still present
    assert cfg["embedding"]["model"] == "gemini-embedding-2-preview"


def test_load_config_data_dir(tmp_path):
    """Config resolves data_dir relative to project root."""
    cfg = load_config(config_path=tmp_path / "nonexistent.yaml", data_dir=tmp_path / "data")
    assert cfg["data_dir"] == str(tmp_path / "data")
