import ast
import inspect
from pathlib import Path

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


def test_reindex_and_update_call_same_rebuild_functions():
    """The update command's reindex step must call every rebuild_* function
    that the reindex command calls. This catches missing imports/calls
    that the formatter strips or that get forgotten when adding new functions."""
    import gmail_search.cli

    cli_source = Path(inspect.getfile(gmail_search.cli)).read_text()
    tree = ast.parse(cli_source)

    def find_rebuild_calls_in_function(func_name):
        """Find all rebuild_* function calls inside a given function."""
        calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        name = ""
                        if isinstance(child.func, ast.Name):
                            name = child.func.id
                        elif isinstance(child.func, ast.Attribute):
                            name = child.func.attr
                        if name.startswith("rebuild_"):
                            calls.add(name)
        return calls

    reindex_calls = find_rebuild_calls_in_function("reindex")
    update_calls = find_rebuild_calls_in_function("update")

    # Every rebuild function in reindex should also be in update
    # (except rebuild_term_aliases which is intentionally slow and skipped in batch updates)
    required_in_update = reindex_calls - {"rebuild_term_aliases"}
    missing = required_in_update - update_calls
    assert not missing, (
        f"update command is missing these rebuild functions that reindex calls: {missing}. "
        f"reindex calls: {reindex_calls}, update calls: {update_calls}"
    )
