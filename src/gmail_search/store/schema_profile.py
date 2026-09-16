"""Which mailbox schema shape this process is talking to.

Callers must not work this out for themselves. Inferring the shape from the
catalog — "does `search_id` exist?" — means every call site can reach a
different answer, and a half-applied migration silently splits the application
in two. So the shape is *selected* once from trusted configuration, carried on
the connection, and verified against the catalog before any mailbox work.

A mismatch raises. There is deliberately no fallback and no retry with another
shape: a reader that quietly downgrades is how a migration turns into a silent
outage rather than a loud failure.
"""
from dataclasses import dataclass
import os
import threading


class SchemaProfileMismatch(RuntimeError):
    """The database does not have the shape this process was told to expect."""


@dataclass(frozen=True)
class SchemaProfile:
    """The parts of the mailbox shape that callers actually branch on."""

    name: str
    message_bm25_key: str
    partitioned: bool
    # `attachments` is keyed on its own PK in every profile — only the message
    # key moves. Keeping it here means callers never fall back to a literal.
    attachment_bm25_key: str = "id"


# The live schema and the TEXT owner-partition target both key BM25 on `id`.
# `numeric-key-v1` is retained only to name the abandoned `search_id` shape, so
# a database still carrying it is reported precisely rather than as "unknown".
TEXT_KEY_V1 = SchemaProfile(name="text-key-v1", message_bm25_key="id", partitioned=False)
TEXT_PARTITIONED_V1 = SchemaProfile(name="text-partitioned-v1", message_bm25_key="id", partitioned=True)
NUMERIC_KEY_V1 = SchemaProfile(name="numeric-key-v1", message_bm25_key="search_id", partitioned=False)

PROFILES = {profile.name: profile for profile in (TEXT_KEY_V1, TEXT_PARTITIONED_V1, NUMERIC_KEY_V1)}
DEFAULT_PROFILE = TEXT_KEY_V1
SELECTION_ENV = "GMS_SCHEMA_PROFILE"

_verified: set[tuple[str, str]] = set()
_lock = threading.Lock()


def selected_profile() -> SchemaProfile:
    """The profile this process was configured with, never one it guessed."""
    name = os.environ.get(SELECTION_ENV)
    if not name:
        return DEFAULT_PROFILE
    try:
        return PROFILES[name]
    except KeyError:
        raise SchemaProfileMismatch(
            f"{SELECTION_ENV}={name!r} is not a known schema profile; "
            f"expected one of {sorted(PROFILES)}"
        ) from None


def bm25_key(conn, table: str = "messages") -> str:
    """The BM25 key field to score on for `table`, per the bound profile.

    Falls back to the default only for connections that carry no profile — raw
    psycopg handles, mostly in tests. Callers must use this rather than a local
    literal, so readers and writers move together across a profile change.
    """
    profile = getattr(conn, "profile", None) or DEFAULT_PROFILE
    if table == "messages":
        return profile.message_bm25_key
    if table == "attachments":
        return profile.attachment_bm25_key
    raise SchemaProfileMismatch(f"no BM25 key is defined for table {table!r}")


def observed_shape(conn) -> tuple[str, bool]:
    """Read the catalog's actual shape: (bm25 key field, partitioned)."""
    row = conn.execute(
        """SELECT (SELECT c.reloptions::text FROM pg_class c JOIN pg_index i ON i.indexrelid=c.oid
                   JOIN pg_am am ON am.oid=c.relam
                   WHERE i.indrelid=to_regclass('messages') AND am.amname='bm25' LIMIT 1),
                  (SELECT c.relkind FROM pg_class c WHERE c.oid=to_regclass('messages'))"""
    ).fetchone()
    reloptions, relkind = (row[0], row[1]) if row else (None, None)
    key = "id"
    if reloptions and "key_field=" in reloptions:
        key = reloptions.split("key_field=", 1)[1].strip("{}\"' ").split(",")[0].strip("\"' ")
    return key, relkind == "p"


def verify(conn, profile: SchemaProfile, *, cache_key: str | None = None) -> None:
    """Fail loudly when the catalog disagrees with the selected profile.

    Verified once per (process, cache_key) because the ~60 call sites open and
    close connections freely and the shape cannot change under a running process
    without a deploy. Pass `cache_key=None` to force a check.
    """
    if cache_key is not None:
        token = (cache_key, profile.name)
        with _lock:
            if token in _verified:
                return
    key, partitioned = observed_shape(conn)
    if key != profile.message_bm25_key or partitioned != profile.partitioned:
        raise SchemaProfileMismatch(
            f"selected profile {profile.name!r} expects "
            f"key_field={profile.message_bm25_key!r} partitioned={profile.partitioned}, "
            f"but the database has key_field={key!r} partitioned={partitioned}"
        )
    if cache_key is not None:
        with _lock:
            _verified.add((cache_key, profile.name))


def reset_verification_cache() -> None:
    """Tests only: forget what has been verified in this process."""
    with _lock:
        _verified.clear()
