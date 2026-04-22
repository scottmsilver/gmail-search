"""Tests for the daemon-dedup guard used by `gmail-search supervise`.

The /proc walker (`is_daemon_running`) delegates its matching logic to
`argv_matches_job`, which is a pure function — so we exhaustively test
that here and only smoke-test the walker end-to-end.
"""

from __future__ import annotations


def test_argv_match_via_console_script():
    from gmail_search.jobs import argv_matches_job

    # `which gmail-search` path — this is how spawn_detached invokes it
    # by default (see gmail_search_command()).
    argv = ["/home/user/anaconda3/bin/gmail-search", "summarize", "--concurrency", "12", "--loop"]
    assert argv_matches_job(argv, "summarize")
    assert not argv_matches_job(argv, "update")
    assert not argv_matches_job(argv, "watch")


def test_argv_match_via_python_module():
    """`python -m gmail_search.cli <cmd>` is the fallback when the
    console script isn't on PATH. Supervise must match this form too.
    """
    from gmail_search.jobs import argv_matches_job

    argv = [
        "/usr/bin/python3",
        "-m",
        "gmail_search.cli",
        "summarize",
        "--loop",
    ]
    assert argv_matches_job(argv, "summarize")
    assert not argv_matches_job(argv, "update")


def test_argv_non_gmail_search_process_never_matches():
    """A shell that happens to have `summarize` in its argv (e.g. a
    pgrep -f gmail-search summarize) must NOT be treated as the daemon.
    """
    from gmail_search.jobs import argv_matches_job

    # The shell's argv has `summarize` as a pattern arg, not as the
    # gmail-search subcommand. Our matcher should see the first non-skip
    # token is "pgrep", not "summarize".
    argv = ["/bin/bash", "-c", "pgrep -f 'gmail-search summarize'"]
    # Our matcher sees argv[0] = "/bin/bash" (first non-skip) → "bash"
    # doesn't equal "summarize". Correct behavior.
    assert not argv_matches_job(argv, "summarize")


def test_argv_empty_or_malformed():
    from gmail_search.jobs import argv_matches_job

    assert not argv_matches_job([], "summarize")
    assert not argv_matches_job(["python"], "summarize")


def test_is_daemon_running_for_nonexistent_job(tmp_path):
    """Smoke test against real /proc. A made-up job id must return False
    because no process will ever have 'wombat_frobnicator' in its argv.
    """
    from gmail_search.jobs import is_daemon_running

    assert is_daemon_running("wombat_frobnicator") is False
