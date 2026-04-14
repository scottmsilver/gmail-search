from click.testing import CliRunner

from gmail_search.cli import main


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Gmail Search" in result.output


def test_cli_status(tmp_path):
    runner = CliRunner()
    result = runner.invoke(main, ["status", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "Messages:" in result.output


def test_cli_cost_empty(tmp_path):
    runner = CliRunner()
    result = runner.invoke(main, ["cost", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "$0.00" in result.output
