"""Strict, side-effect-free production configuration for invited-user runtimes."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping
import unicodedata

from .auth.invited_broker import BoundGmailBroker
from .gateway.database import ReaderCredential
from .gateway.maintenance import ReleaseIdentity
from .gateway.partition_profiles import TEXT_OWNER_PARTITIONS_V1
from .gateway.provider import ProviderProfile
from .gateway.search_embedding import GeminiEmbeddingProfile
from .gateway.search_index import IndexBinding
from .gateway.search_reader import SearchCredential, SearchProfile
from .gateway.search_reranker import GeminiRerankerProfile
from .gateway.writer import WriterCredential


_MAX_CONFIG_BYTES = 1024 * 1024
_MAX_STRING_CHARS = 8192
_MAX_INT = 2**63 - 1
_BUDGET_ID = re.compile(r"[0-9a-f]{32}").fullmatch
_ERROR = "Runtime configuration is invalid."


class ConfigError(ValueError):
    """A deliberately non-diagnostic error safe for production logs."""


@dataclass(frozen=True)
class IndexConfig:
    path: Path
    generation: str
    source_id: str


@dataclass(frozen=True)
class OwnerConfig:
    id: str
    email: str
    google_subject: str = field(repr=False)
    reader_dsn: str = field(repr=False)
    search_dsn: str = field(repr=False)
    writer_dsn: str = field(repr=False)
    budget_id: str
    index: IndexConfig


@dataclass(frozen=True)
class WorkerConfig:
    host: str
    port: int
    private_key: Path
    known_hosts: Path


@dataclass(frozen=True)
class BrokerConfig:
    origin: str
    bearer: str = field(repr=False)
    signing_secret: str = field(repr=False)


@dataclass(frozen=True)
class ProviderConfig:
    anthropic_key: str = field(repr=False)
    gemini_key: str = field(repr=False)
    input_units_per_token: int
    output_units_per_token: int
    embedding_units_per_token: int
    rerank_input_units_per_token: int
    rerank_output_units_per_token: int
    fact_model_tag: str


@dataclass(frozen=True)
class ReleaseConfig:
    store_id: str
    release_epoch: int


@dataclass(frozen=True)
class RuntimeConfig:
    version: int
    state_dir: Path
    attachment_root: Path
    release: ReleaseConfig
    worker: WorkerConfig
    broker: BrokerConfig
    provider: ProviderConfig
    owners: tuple[OwnerConfig, ...]


def _invalid() -> ConfigError:
    return ConfigError(_ERROR)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _invalid()
        result[key] = value
    return result


def _reject_constant(_value):
    raise _invalid()


def _read_private_file(path: os.PathLike[str] | str) -> bytes:
    try:
        source = Path(path)
    except (TypeError, ValueError, OSError):
        raise _invalid() from None
    if not source.is_absolute() or not hasattr(os, "O_NOFOLLOW"):
        raise _invalid()
    descriptor = None
    try:
        descriptor = os.open(
            source,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_uid != os.getuid()
            or before.st_size > _MAX_CONFIG_BYTES
        ):
            raise _invalid()
        chunks = []
        remaining = _MAX_CONFIG_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            len(raw) > _MAX_CONFIG_BYTES
            or not stat.S_ISREG(after.st_mode)
            or stat.S_IMODE(after.st_mode) != 0o600
            or after.st_uid != os.getuid()
            or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        ):
            raise _invalid()
        return raw
    except ConfigError:
        raise
    except (OSError, OverflowError):
        raise _invalid() from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                raise _invalid() from None


def _object(value: Any, fields: set[str]) -> Mapping[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise _invalid()
    return value


def _string(value: Any) -> str:
    if (
        type(value) is not str
        or not value
        or not value.strip()
        or len(value) > _MAX_STRING_CHARS
        or any(unicodedata.category(character).startswith("C") for character in value)
    ):
        raise _invalid()
    return value


def _positive_int(value: Any, *, maximum: int = _MAX_INT) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise _invalid()
    return value


def _absolute_path(value: Any) -> Path:
    path = Path(_string(value))
    if not path.is_absolute() or ".." in path.parts:
        raise _invalid()
    return path


def _owner_id(value: Any) -> str:
    value = _string(value)
    if len(value) > 512 or any(character.isspace() or ord(character) < 33 for character in value):
        raise _invalid()
    return value


def _email(value: Any) -> str:
    value = _string(value)
    if (
        not 3 <= len(value) <= 254
        or value != value.strip().lower()
        or value.count("@") != 1
        or any(character.isspace() or ord(character) < 33 for character in value)
    ):
        raise _invalid()
    local, domain = value.split("@")
    if not local or not domain:
        raise _invalid()
    return value


def _google_subject(value: Any) -> str:
    value = _string(value)
    if len(value) > 255 or any(character.isspace() or ord(character) < 33 for character in value):
        raise _invalid()
    return value


def _private_directory(value: Any) -> Path:
    path = _absolute_path(value)
    descriptor = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY,
        )
        info = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o700
            or info.st_uid != os.getuid()
        ):
            raise _invalid()
    except ConfigError:
        raise
    except (OSError, OverflowError):
        raise _invalid() from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                raise _invalid() from None
    return path


def _release(value: Any) -> ReleaseConfig:
    value = _object(value, {"store_id", "release_epoch"})
    store_id = _string(value["store_id"])
    release_epoch = _positive_int(value["release_epoch"])
    ReleaseIdentity(store_id, TEXT_OWNER_PARTITIONS_V1, release_epoch)
    return ReleaseConfig(store_id=store_id, release_epoch=release_epoch)


def _worker(value: Any) -> WorkerConfig:
    value = _object(value, {"host", "port", "private_key", "known_hosts"})
    return WorkerConfig(
        host=_string(value["host"]),
        port=_positive_int(value["port"], maximum=65535),
        private_key=_absolute_path(value["private_key"]),
        known_hosts=_absolute_path(value["known_hosts"]),
    )


def _broker(value: Any) -> BrokerConfig:
    value = _object(value, {"origin", "bearer", "signing_secret"})
    origin = _string(value["origin"])
    bearer = _string(value["bearer"])
    signing_secret = _string(value["signing_secret"])
    validation_client = type("_ValidationClient", (), {"trust_env": False})()
    BoundGmailBroker(
        origin,
        bearer=bearer,
        signing_secret=signing_secret,
        client=validation_client,
    )
    return BrokerConfig(origin=origin, bearer=bearer, signing_secret=signing_secret)


def _provider_secret(value: Any) -> str:
    value = _string(value)
    if len(value) > 512 or not value.isascii() or any(not 33 <= ord(char) <= 126 for char in value):
        raise _invalid()
    return value


def _provider(value: Any) -> ProviderConfig:
    fields = {
        "anthropic_key",
        "gemini_key",
        "input_units_per_token",
        "output_units_per_token",
        "embedding_units_per_token",
        "rerank_input_units_per_token",
        "rerank_output_units_per_token",
        "fact_model_tag",
    }
    value = _object(value, fields)
    anthropic_key = _provider_secret(value["anthropic_key"])
    gemini_key = _provider_secret(value["gemini_key"])
    input_rate = _positive_int(value["input_units_per_token"], maximum=10**8)
    output_rate = _positive_int(value["output_units_per_token"], maximum=10**8)
    embedding_rate = _positive_int(value["embedding_units_per_token"], maximum=10**8)
    rerank_input_rate = _positive_int(value["rerank_input_units_per_token"], maximum=10**8)
    rerank_output_rate = _positive_int(value["rerank_output_units_per_token"], maximum=10**8)
    fact_model_tag = _string(value["fact_model_tag"])
    ProviderProfile(
        model="claude-sonnet-4-6",
        input_token_limit=200000,
        output_token_limit=4096,
        input_units_per_token=input_rate,
        output_units_per_token=output_rate,
        client_profile="pi-0.84.4",
        effort="high",
    )
    GeminiEmbeddingProfile(embedding_rate)
    GeminiRerankerProfile(
        rerank_input_rate,
        rerank_output_rate,
        "full-model-ceilings-v1",
    )
    SearchProfile(
        "gemini-embedding-2",
        fact_model_tag,
        3072,
        schema_profile=TEXT_OWNER_PARTITIONS_V1,
    )
    return ProviderConfig(
        anthropic_key=anthropic_key,
        gemini_key=gemini_key,
        input_units_per_token=input_rate,
        output_units_per_token=output_rate,
        embedding_units_per_token=embedding_rate,
        rerank_input_units_per_token=rerank_input_rate,
        rerank_output_units_per_token=rerank_output_rate,
        fact_model_tag=fact_model_tag,
    )


def _index(value: Any, owner_id: str) -> IndexConfig:
    value = _object(value, {"path", "generation", "source_id"})
    path = _absolute_path(value["path"])
    generation = _string(value["generation"])
    source_id = _string(value["source_id"])
    IndexBinding(owner_id, generation, "gemini-embedding-2", 3072, source_id)
    return IndexConfig(path=path, generation=generation, source_id=source_id)


def _owner(value: Any) -> OwnerConfig:
    fields = {
        "id",
        "email",
        "google_subject",
        "reader_dsn",
        "search_dsn",
        "writer_dsn",
        "budget_id",
        "index",
    }
    value = _object(value, fields)
    owner_id = _owner_id(value["id"])
    email = _email(value["email"])
    google_subject = _google_subject(value["google_subject"])
    reader_dsn = _string(value["reader_dsn"])
    search_dsn = _string(value["search_dsn"])
    writer_dsn = _string(value["writer_dsn"])
    budget_id = _string(value["budget_id"])
    if _BUDGET_ID(budget_id) is None:
        raise _invalid()
    try:
        ReaderCredential(owner_id, reader_dsn)
        SearchCredential(owner_id, search_dsn, schema_profile=TEXT_OWNER_PARTITIONS_V1)
        WriterCredential(owner_id, writer_dsn)
    except Exception:
        raise _invalid() from None
    return OwnerConfig(
        id=owner_id,
        email=email,
        google_subject=google_subject,
        reader_dsn=reader_dsn,
        search_dsn=search_dsn,
        writer_dsn=writer_dsn,
        budget_id=budget_id,
        index=_index(value["index"], owner_id),
    )


def _owners(value: Any) -> tuple[OwnerConfig, ...]:
    if type(value) is not list or not value:
        raise _invalid()
    owners = tuple(_owner(item) for item in value)
    if (
        len({owner.id for owner in owners}) != len(owners)
        or len({owner.email.casefold() for owner in owners}) != len(owners)
        or len({owner.google_subject for owner in owners}) != len(owners)
    ):
        raise _invalid()
    return owners


def _build(value: Any) -> RuntimeConfig:
    fields = {
        "version",
        "state_dir",
        "attachment_root",
        "release",
        "worker",
        "broker",
        "provider",
        "owners",
    }
    value = _object(value, fields)
    if type(value["version"]) is not int or value["version"] != 1:
        raise _invalid()
    return RuntimeConfig(
        version=1,
        state_dir=_private_directory(value["state_dir"]),
        attachment_root=_absolute_path(value["attachment_root"]),
        release=_release(value["release"]),
        worker=_worker(value["worker"]),
        broker=_broker(value["broker"]),
        provider=_provider(value["provider"]),
        owners=_owners(value["owners"]),
    )


def load_runtime_config(path: os.PathLike[str] | str) -> RuntimeConfig:
    """Load one immutable configuration without creating or changing resources."""

    try:
        raw = _read_private_file(path)
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        return _build(value)
    except Exception:
        raise _invalid() from None


__all__ = [
    "BrokerConfig",
    "ConfigError",
    "IndexConfig",
    "OwnerConfig",
    "ProviderConfig",
    "ReleaseConfig",
    "RuntimeConfig",
    "WorkerConfig",
    "load_runtime_config",
]
