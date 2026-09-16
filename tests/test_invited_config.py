import json
import os
from dataclasses import FrozenInstanceError

import pytest

from gmail_search.gateway.database import reader_role
from gmail_search.gateway.search_reader import search_role
from gmail_search.gateway.partition_profiles import TEXT_OWNER_PARTITIONS_V1
from gmail_search.gateway.writer import application_writer_role
from gmail_search.invited_config import ConfigError, load_runtime_config


SECRETS = (
    "reader-password-secret",
    "search-password-secret",
    "writer-password-secret",
    "google-subject-secret",
    "broker-bearer-secret-0123456789abcdef",
    "broker-signing-secret-fedcba9876543210",
    "anthropic-key-secret",
    "gemini-key-secret",
)


def _dsn(role, password):
    return f"dbname=mail user={role} password={password} host=127.0.0.1"


def _value(tmp_path):
    state = tmp_path / "state"
    state.mkdir(mode=0o700, exist_ok=True)
    state.chmod(0o700)
    owner = "owner-a"
    return {
        "version": 1,
        "state_dir": str(state),
        "attachment_root": str(tmp_path / "attachments"),
        "release": {"store_id": "mail-store", "release_epoch": 7},
        "worker": {
            "host": "worker.internal",
            "port": 22092,
            "private_key": str(tmp_path / "id_ed25519"),
            "known_hosts": str(tmp_path / "known_hosts"),
        },
        "broker": {
            "origin": "https://broker.internal",
            "bearer": "broker-bearer-secret-0123456789abcdef",
            "signing_secret": "broker-signing-secret-fedcba9876543210",
        },
        "provider": {
            "anthropic_key": "anthropic-key-secret",
            "gemini_key": "gemini-key-secret",
            "input_units_per_token": 2,
            "output_units_per_token": 10,
            "embedding_units_per_token": 1,
            "rerank_input_units_per_token": 3,
            "rerank_output_units_per_token": 12,
            "fact_model_tag": "gemini-2.5-flash-facts-v1",
        },
        "owners": [{
            "id": owner,
            "email": "owner@example.com",
            "google_subject": "google-subject-secret",
            "reader_dsn": _dsn(reader_role(owner), "reader-password-secret"),
            "search_dsn": _dsn(
                search_role(owner, profile=TEXT_OWNER_PARTITIONS_V1),
                "search-password-secret",
            ),
            "writer_dsn": _dsn(application_writer_role(owner), "writer-password-secret"),
            "budget_id": "0123456789abcdef0123456789abcdef",
            "index": {
                "path": str(tmp_path / "indexes" / owner),
                "generation": "generation-4",
                "source_id": "qualified-index-v4",
            },
        }],
    }


def _write(tmp_path, value, *, name="runtime.json", mode=0o600):
    path = tmp_path / name
    path.write_text(json.dumps(value))
    path.chmod(mode)
    return path


def _assert_secret_free(exc):
    rendered = f"{exc!s} {exc!r}"
    assert all(secret not in rendered for secret in SECRETS)


def test_loads_frozen_typed_configuration_without_exposing_secrets(tmp_path):
    config = load_runtime_config(_write(tmp_path, _value(tmp_path)))

    assert config.version == 1
    assert config.state_dir == tmp_path / "state"
    assert config.worker.private_key == tmp_path / "id_ed25519"
    assert config.provider.rerank_output_units_per_token == 12
    assert isinstance(config.owners, tuple)
    assert config.owners[0].index.generation == "generation-4"
    with pytest.raises(FrozenInstanceError):
        config.version = 2
    rendered = repr(config)
    assert all(secret not in rendered for secret in SECRETS)


@pytest.mark.parametrize("mode", [0o400, 0o640, 0o660, 0o700])
def test_rejects_config_without_exact_0600_mode(tmp_path, mode):
    with pytest.raises(ConfigError) as caught:
        load_runtime_config(_write(tmp_path, _value(tmp_path), mode=mode))
    _assert_secret_free(caught.value)


def test_rejects_relative_config_path(tmp_path, monkeypatch):
    path = _write(tmp_path, _value(tmp_path))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError):
        load_runtime_config(path.name)


def test_rejects_config_symlink(tmp_path):
    target = _write(tmp_path, _value(tmp_path), name="target.json")
    link = tmp_path / "runtime.json"
    link.symlink_to(target)
    with pytest.raises(ConfigError):
        load_runtime_config(link)


def test_config_open_is_nonblocking_and_does_not_follow(tmp_path, monkeypatch):
    path = _write(tmp_path, _value(tmp_path))
    real_open = os.open
    flags_seen = []

    def recording_open(target, flags, *args):
        flags_seen.append(flags)
        return real_open(target, flags, *args)

    monkeypatch.setattr(os, "open", recording_open)
    load_runtime_config(path)
    assert flags_seen[0] & os.O_NONBLOCK
    assert flags_seen[0] & os.O_NOFOLLOW


def test_rejects_oversized_config_before_json_decoding(tmp_path):
    path = tmp_path / "runtime.json"
    path.write_bytes(b" " * (1024 * 1024 + 1))
    path.chmod(0o600)
    with pytest.raises(ConfigError):
        load_runtime_config(path)


@pytest.mark.parametrize("mode", [0o600, 0o755, 0o770])
def test_rejects_state_directory_without_exact_0700_mode(tmp_path, mode):
    value = _value(tmp_path)
    (tmp_path / "state").chmod(mode)
    with pytest.raises(ConfigError):
        load_runtime_config(_write(tmp_path, value))


def test_rejects_state_directory_symlink(tmp_path):
    value = _value(tmp_path)
    real = tmp_path / "real-state"
    real.mkdir(mode=0o700)
    (tmp_path / "state").rmdir()
    (tmp_path / "state").symlink_to(real, target_is_directory=True)
    with pytest.raises(ConfigError):
        load_runtime_config(_write(tmp_path, value))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update({"unexpected": "field"}),
        lambda value: value["worker"].update({"unexpected": "field"}),
        lambda value: value["owners"][0]["index"].update({"unexpected": "field"}),
        lambda value: value["provider"].pop("fact_model_tag"),
    ],
)
def test_rejects_unknown_or_missing_keys(tmp_path, mutate):
    value = _value(tmp_path)
    mutate(value)
    with pytest.raises(ConfigError):
        load_runtime_config(_write(tmp_path, value))


def test_rejects_duplicate_json_keys(tmp_path):
    value = _value(tmp_path)
    raw = json.dumps(value)
    raw = raw.replace('"version": 1', '"version": 1, "version": 1', 1)
    path = tmp_path / "runtime.json"
    path.write_text(raw)
    path.chmod(0o600)
    with pytest.raises(ConfigError):
        load_runtime_config(path)


@pytest.mark.parametrize("field", ["id", "email", "google_subject"])
def test_rejects_duplicate_owner_identities(tmp_path, field):
    value = _value(tmp_path)
    second = dict(value["owners"][0])
    second["index"] = dict(second["index"])
    second.update({"id": "owner-b", "email": "second@example.com", "google_subject": "sub-b"})
    second["reader_dsn"] = _dsn(reader_role("owner-b"), "reader-b")
    second["search_dsn"] = _dsn(search_role("owner-b", profile=TEXT_OWNER_PARTITIONS_V1), "search-b")
    second["writer_dsn"] = _dsn(application_writer_role("owner-b"), "writer-b")
    second[field] = value["owners"][0][field]
    value["owners"].append(second)
    with pytest.raises(ConfigError):
        load_runtime_config(_write(tmp_path, value))


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_units_per_token", 0),
        ("output_units_per_token", -1),
        ("embedding_units_per_token", True),
        ("rerank_input_units_per_token", 1.5),
        ("rerank_output_units_per_token", False),
    ],
)
def test_rejects_non_positive_integer_provider_costs(tmp_path, field, value):
    config = _value(tmp_path)
    config["provider"][field] = value
    with pytest.raises(ConfigError):
        load_runtime_config(_write(tmp_path, config))


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_units_per_token", 10**8),
        ("embedding_units_per_token", 10**8 + 1),
        ("rerank_input_units_per_token", 10**8 + 1),
        ("rerank_output_units_per_token", 10**8 + 1),
    ],
)
def test_rejects_provider_rates_outside_fixed_profile_reservations(tmp_path, field, value):
    config = _value(tmp_path)
    config["provider"][field] = value
    with pytest.raises(ConfigError):
        load_runtime_config(_write(tmp_path, config))


@pytest.mark.parametrize("key", ["anthropic_key", "gemini_key"])
def test_rejects_provider_keys_outside_printable_ascii_contract(tmp_path, key):
    value = _value(tmp_path)
    value["provider"][key] = "x" * 513
    with pytest.raises(ConfigError):
        load_runtime_config(_write(tmp_path, value))

    value = _value(tmp_path)
    value["provider"][key] = "snowman-☃"
    with pytest.raises(ConfigError):
        load_runtime_config(_write(tmp_path, value, name=f"bad-{key}.json"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("bearer", "short"),
        ("signing_secret", "contains space " + "x" * 32),
        ("signing_secret", "broker-bearer-secret-0123456789abcdef"),
    ],
)
def test_rejects_broker_values_outside_bound_broker_contract(tmp_path, field, value):
    config = _value(tmp_path)
    config["broker"][field] = value
    with pytest.raises(ConfigError):
        load_runtime_config(_write(tmp_path, config))


@pytest.mark.parametrize("generation", [4, "", "x" * 129])
def test_rejects_invalid_string_index_generation(tmp_path, generation):
    config = _value(tmp_path)
    config["owners"][0]["index"]["generation"] = generation
    with pytest.raises(ConfigError):
        load_runtime_config(_write(tmp_path, config))


@pytest.mark.parametrize("store_id", ["has space", "x" * 129])
def test_rejects_release_identity_not_accepted_by_gate(tmp_path, store_id):
    config = _value(tmp_path)
    config["release"]["store_id"] = store_id
    with pytest.raises(ConfigError):
        load_runtime_config(_write(tmp_path, config))


@pytest.mark.parametrize("dsn_field", ["reader_dsn", "search_dsn", "writer_dsn"])
def test_rejects_dsn_for_wrong_fixed_owner_role_without_leaking_it(tmp_path, dsn_field):
    value = _value(tmp_path)
    value["owners"][0][dsn_field] = "user=wrong password=dsn-leak-secret"
    with pytest.raises(ConfigError) as caught:
        load_runtime_config(_write(tmp_path, value))
    assert "dsn-leak-secret" not in f"{caught.value!s} {caught.value!r}"


def test_rejects_control_characters_and_non_hex_budget_ids(tmp_path):
    value = _value(tmp_path)
    value["worker"]["host"] = "worker\ninternal"
    with pytest.raises(ConfigError):
        load_runtime_config(_write(tmp_path, value))

    value = _value(tmp_path)
    value["owners"][0]["budget_id"] = "not-a-registry-budget"
    with pytest.raises(ConfigError):
        load_runtime_config(_write(tmp_path, value, name="bad-budget.json"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", "owner with space"),
        ("id", "x" * 513),
        ("email", "Owner@Example.com"),
        ("email", "not-an-email"),
        ("google_subject", "subject with space"),
        ("google_subject", "x" * 256),
    ],
)
def test_rejects_owner_identity_values_the_identity_store_cannot_bind(tmp_path, field, value):
    config = _value(tmp_path)
    config["owners"][0][field] = value
    with pytest.raises(ConfigError):
        load_runtime_config(_write(tmp_path, config))


def test_rejects_parent_components_in_runtime_paths(tmp_path):
    config = _value(tmp_path)
    config["attachment_root"] = str(tmp_path / "nested" / ".." / "attachments")
    with pytest.raises(ConfigError):
        load_runtime_config(_write(tmp_path, config))


def test_rejects_wrong_file_owner(tmp_path, monkeypatch):
    path = _write(tmp_path, _value(tmp_path))
    monkeypatch.setattr(os, "getuid", lambda: path.stat().st_uid + 1)
    with pytest.raises(ConfigError):
        load_runtime_config(path)
