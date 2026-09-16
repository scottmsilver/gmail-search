import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('provision_public', Path(__file__).parents[1] / 'deploy/public/provision_database.py')
provision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(provision)


def test_owner_policy_is_fixed_and_restrictive():
    sql = provision.build_sql("owner'quote", 'gmail_search')
    assert 'current_setting' not in sql
    assert 'AS RESTRICTIVE FOR ALL TO "gmail_search_public"' in sql
    assert "user_id::text = 'owner''quote'" in sql
    for table in provision.ALL_TABLES:
        assert f'ALTER TABLE public."{table}" ENABLE ROW LEVEL SECURITY;' in sql
    assert 'NOBYPASSRLS' in sql and 'NOINHERIT' in sql and 'NOSUPERUSER' in sql
    assert 'PASSWORD' not in sql


def test_grants_exclude_mail_writes_and_global_state():
    sql = provision.build_sql('owner', 'db')
    assert 'GRANT SELECT ON public."messages"' in sql
    assert 'GRANT SELECT ON public."users"' in sql
    assert 'GRANT SELECT, INSERT, UPDATE, DELETE ON public."conversations"' in sql
    for table in ('job_progress', 'query_cache', 'sync_state', 'mcp_oauth_state', 'invited_emails'):
        assert f'public."{table}"' not in sql
    assert 'GRANT USAGE ON SEQUENCE public."agent_events_id_seq"' in sql
    assert 'GRANT USAGE ON SEQUENCE public."agent_artifacts_id_seq"' not in sql


def test_child_ownership_joins():
    assert 'public.conversations c' in provision.owner_predicate('conversation_messages', 'owner')
    assert 'public.agent_sessions s' in provision.owner_predicate('agent_events', 'owner')
    assert 'public.agent_sessions s' in provision.owner_predicate('agent_artifacts', 'owner')
    assert provision.owner_predicate('users', 'owner') == "id::text = 'owner'"


def test_legacy_policy_preserves_only_explicit_roles():
    sql = provision.build_sql('owner', 'db', {'embeddings': ['gmail_analyst']})
    assert 'FOR ALL TO "gmail_analyst" USING (true)' in sql
    assert 'TO PUBLIC USING (true)' not in sql
    assert 'GRANT SELECT ON public."embeddings" TO "gmail_analyst"' not in sql


def test_password_is_reused_and_private(tmp_path):
    path = tmp_path / 'password'
    first = provision.password_file(path)
    assert len(first) >= 43
    assert path.stat().st_mode & 0o777 == 0o600
    assert provision.password_file(path) == first
    path.chmod(0o644)
    with pytest.raises(ValueError):
        provision.password_file(path)


def test_password_symlink_rejected(tmp_path):
    target = tmp_path / 'target'
    target.write_text('x' * 64)
    link = tmp_path / 'link'
    link.symlink_to(target)
    with pytest.raises(OSError):
        provision.password_file(link)
    assert target.read_text() == 'x' * 64
