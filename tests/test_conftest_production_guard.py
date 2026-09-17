"""The test suite must be unable to reach the production mailbox.

Twice now a test has written into the live database. The first time it left 28
`test_<uuid8>` schemas, 326 tables and 87 MB. The second time it repointed the
production owner's ScaNN index at a pytest temp directory, which silently
disabled semantic search for the main mailbox until someone looked.

Both had the same shape: the isolation fixture pins `search_path` to
`<schema>,public`, so a relation missing from the disposable schema resolves to
`public` instead of failing. That fallback is only dangerous because the base
DSN *can* be the production cluster. Close that, and the fallback stops mattering.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def _conftest():
    spec = importlib.util.spec_from_file_location('gms_conftest', ROOT / 'tests/conftest.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('dsn', [
    'host=127.0.0.1 port=5544 dbname=gmail_search user=gmail_search',
    'postgresql://gmail_search:gmail_search@127.0.0.1:5544/gmail_search',
    'host=/var/run/postgresql dbname=gmail_search',
])
def test_the_production_database_is_refused_as_a_test_target(dsn):
    module = _conftest()
    with pytest.raises(RuntimeError, match='production'):
        module.reject_production_dsn(dsn)


@pytest.mark.parametrize('dsn', [
    'host=127.0.0.1 port=55440 dbname=postgres user=postgres',
    'host=127.0.0.1 port=5544 dbname=gms_disposable user=postgres',
    '',
])
def test_disposable_targets_are_accepted(dsn):
    _conftest().reject_production_dsn(dsn)


def test_the_guard_runs_at_import_not_only_on_request():
    """A guard nobody calls is not a guard. Importing conftest with a production
    DSN configured must fail the run outright."""
    source = (ROOT / 'tests/conftest.py').read_text()
    assert 'reject_production_dsn(_PG_BASE_DSN)' in source, \
        'the guard must be applied to the configured DSN at module import'
