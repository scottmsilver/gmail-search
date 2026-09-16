from __future__ import annotations

import asyncio
import inspect
import signal
from contextlib import contextmanager

import pytest


class _FakeListener:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeServer:
    def __init__(self, config) -> None:
        self.config = config
        self.should_exit = False
        self.started = False
        self.start_event = asyncio.Event()
        self.drained = False
        self.sockets = None

    async def serve(self, sockets=None) -> None:
        self.sockets = sockets
        self.started = True
        self.start_event.set()
        while not self.should_exit:
            await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.drained = True


class _SignalOwner:
    def __init__(self) -> None:
        self.trigger = None
        self.entered = False
        self.exited = False

    @contextmanager
    def manage(self, trigger):
        self.trigger = trigger
        self.entered = True
        try:
            yield
        finally:
            self.exited = True


def test_server_configs_are_fixed_private_single_worker_listeners():
    from gmail_search.invited_listeners import _build_server_config

    browser_app = object()
    gateway_app = object()

    browser = _build_server_config(browser_app, port=8091)
    gateway = _build_server_config(gateway_app, port=8092)

    assert browser.app is browser_app
    assert gateway.app is gateway_app
    assert browser.host == gateway.host == "127.0.0.1"
    assert (browser.port, gateway.port) == (8091, 8092)
    assert browser.access_log is gateway.access_log is False
    # Uvicorn's lifespan logger includes the raw exception text and traceback;
    # caller-level startup reporting must remain generic so secrets cannot leak.
    assert browser.log_level == gateway.log_level == "critical"
    assert browser.reload is gateway.reload is False
    assert browser.workers == gateway.workers == 1


def test_public_api_does_not_expose_binding_or_runtime_overrides():
    from gmail_search.invited_listeners import serve_apps

    assert list(inspect.signature(serve_apps).parameters) == [
        "browser_app",
        "gateway_app",
    ]


@pytest.mark.asyncio
async def test_second_bind_failure_closes_first_before_apps_start():
    from gmail_search.invited_listeners import _serve_apps

    first_listener = _FakeListener()
    bind_calls: list[tuple[str, int]] = []

    def open_listener(host: str, port: int):
        bind_calls.append((host, port))
        if port == 8092:
            raise OSError("gateway port unavailable")
        return first_listener

    def unexpected_server_factory(config):
        raise AssertionError("apps must not start until both sockets are bound")

    with pytest.raises(OSError, match="gateway port unavailable"):
        await _serve_apps(
            object(),
            object(),
            listener_factory=open_listener,
            server_factory=unexpected_server_factory,
        )

    assert bind_calls == [("127.0.0.1", 8091), ("127.0.0.1", 8092)]
    assert first_listener.closed


@pytest.mark.asyncio
async def test_signal_stops_and_drains_both_servers_then_cleans_up_handlers():
    from gmail_search.invited_listeners import _serve_apps

    browser_app = object()
    gateway_app = object()
    listeners = [_FakeListener(), _FakeListener()]
    servers: list[_FakeServer] = []
    signals = _SignalOwner()

    def open_listener(host: str, port: int):
        assert host == "127.0.0.1"
        return listeners[port - 8091]

    def make_server(config):
        server = _FakeServer(config)
        servers.append(server)
        return server

    serve_task = asyncio.create_task(
        _serve_apps(
            browser_app,
            gateway_app,
            listener_factory=open_listener,
            server_factory=make_server,
            signal_context_factory=signals.manage,
        )
    )
    while len(servers) < 2:
        await asyncio.sleep(0)
    await asyncio.gather(*(server.start_event.wait() for server in servers))

    assert [server.config.app for server in servers] == [browser_app, gateway_app]
    assert [server.sockets for server in servers] == [[listeners[0]], [listeners[1]]]
    assert signals.entered
    assert signals.trigger is not None
    signals.trigger()
    await serve_task

    assert all(server.drained for server in servers)
    assert all(listener.closed for listener in listeners)
    assert signals.exited


@pytest.mark.asyncio
async def test_browser_does_not_start_accepting_until_gateway_is_ready():
    from gmail_search.invited_listeners import _serve_apps

    listeners = [_FakeListener(), _FakeListener()]
    servers: dict[int, _FakeServer] = {}
    gateway_entered = asyncio.Event()
    allow_gateway_start = asyncio.Event()
    signals = _SignalOwner()

    class DelayedGateway(_FakeServer):
        async def serve(self, sockets=None) -> None:
            gateway_entered.set()
            await allow_gateway_start.wait()
            await super().serve(sockets=sockets)

    def make_server(config):
        server = DelayedGateway(config) if config.port == 8092 else _FakeServer(config)
        servers[config.port] = server
        return server

    serve_task = asyncio.create_task(
        _serve_apps(
            object(),
            object(),
            listener_factory=lambda host, port: listeners[port - 8091],
            server_factory=make_server,
            signal_context_factory=signals.manage,
        )
    )
    await gateway_entered.wait()
    await asyncio.sleep(0)

    assert not servers[8091].start_event.is_set()
    allow_gateway_start.set()
    await servers[8091].start_event.wait()
    assert servers[8092].started

    assert signals.trigger is not None
    signals.trigger()
    await serve_task


@pytest.mark.asyncio
async def test_startup_failure_stops_and_awaits_sibling_shutdown():
    from gmail_search.invited_listeners import _serve_apps

    listeners = [_FakeListener(), _FakeListener()]
    signals = _SignalOwner()
    gateway_server: _FakeServer | None = None

    class FailingBrowser(_FakeServer):
        async def serve(self, sockets=None) -> None:
            assert gateway_server is not None
            await gateway_server.start_event.wait()
            raise RuntimeError("provider key sk-secret")

    def make_server(config):
        nonlocal gateway_server
        if config.port == 8092:
            gateway_server = _FakeServer(config)
            return gateway_server
        return FailingBrowser(config)

    with pytest.raises(RuntimeError, match="^browser listener failed$") as raised:
        await _serve_apps(
            object(),
            object(),
            listener_factory=lambda host, port: listeners[port - 8091],
            server_factory=make_server,
            signal_context_factory=signals.manage,
        )

    assert "sk-secret" not in str(raised.value)
    assert gateway_server is not None
    assert gateway_server.drained
    assert all(listener.closed for listener in listeners)
    assert signals.exited


@pytest.mark.asyncio
async def test_startup_system_exit_cannot_skip_sibling_cleanup():
    from gmail_search.invited_listeners import _serve_apps

    listeners = [_FakeListener(), _FakeListener()]
    signals = _SignalOwner()
    gateway_server: _FakeServer | None = None

    class ExitingBrowser(_FakeServer):
        async def serve(self, sockets=None) -> None:
            assert gateway_server is not None
            await gateway_server.start_event.wait()
            raise SystemExit(7)

    def make_server(config):
        nonlocal gateway_server
        if config.port == 8092:
            gateway_server = _FakeServer(config)
            return gateway_server
        return ExitingBrowser(config)

    with pytest.raises(SystemExit) as raised:
        await _serve_apps(
            object(),
            object(),
            listener_factory=lambda host, port: listeners[port - 8091],
            server_factory=make_server,
            signal_context_factory=signals.manage,
        )

    assert raised.value.code == 7
    assert gateway_server is not None
    assert gateway_server.drained
    assert all(listener.closed for listener in listeners)
    assert signals.exited


@pytest.mark.asyncio
async def test_uvicorn_startup_rejection_is_reported_after_sibling_drains():
    from gmail_search.invited_listeners import _serve_apps

    listeners = [_FakeListener(), _FakeListener()]
    gateway_server: _FakeServer | None = None

    class RejectedBrowser(_FakeServer):
        async def serve(self, sockets=None) -> None:
            assert gateway_server is not None
            await gateway_server.start_event.wait()
            # Uvicorn reports ASGI lifespan startup failures by setting
            # should_exit and returning without setting started.
            self.should_exit = True

    def make_server(config):
        nonlocal gateway_server
        if config.port == 8092:
            gateway_server = _FakeServer(config)
            return gateway_server
        return RejectedBrowser(config)

    with pytest.raises(RuntimeError, match="browser listener failed to start"):
        await _serve_apps(
            object(),
            object(),
            listener_factory=lambda host, port: listeners[port - 8091],
            server_factory=make_server,
            signal_context_factory=_SignalOwner().manage,
        )

    assert gateway_server is not None
    assert gateway_server.drained
    assert all(listener.closed for listener in listeners)


@pytest.mark.asyncio
async def test_signal_owner_removes_handlers_and_restores_previous_values(monkeypatch):
    from gmail_search.invited_listeners import _signal_owner

    loop = asyncio.get_running_loop()
    added = []
    removed = []
    restored = []
    previous = {
        signal.SIGINT: object(),
        signal.SIGTERM: object(),
    }

    monkeypatch.setattr(
        loop,
        "add_signal_handler",
        lambda signum, callback: added.append((signum, callback)),
    )
    monkeypatch.setattr(
        loop,
        "remove_signal_handler",
        lambda signum: removed.append(signum),
    )
    monkeypatch.setattr(signal, "getsignal", lambda signum: previous[signum])
    monkeypatch.setattr(
        signal,
        "signal",
        lambda signum, handler: restored.append((signum, handler)),
    )
    trigger = lambda: None

    with _signal_owner(trigger):
        assert added == [(signal.SIGINT, trigger), (signal.SIGTERM, trigger)]

    assert removed == [signal.SIGINT, signal.SIGTERM]
    assert restored == [
        (signal.SIGINT, previous[signal.SIGINT]),
        (signal.SIGTERM, previous[signal.SIGTERM]),
    ]


@pytest.mark.asyncio
async def test_repeated_caller_cancellation_cannot_interrupt_lifespan_drain():
    from gmail_search.invited_listeners import _serve_apps

    listeners = [_FakeListener(), _FakeListener()]
    signals = _SignalOwner()
    servers: list[_FakeServer] = []
    allow_drain = asyncio.Event()

    class SlowDrainServer(_FakeServer):
        def __init__(self, config) -> None:
            super().__init__(config)
            self.drain_started = asyncio.Event()

        async def serve(self, sockets=None) -> None:
            self.sockets = sockets
            self.started = True
            self.start_event.set()
            while not self.should_exit:
                await asyncio.sleep(0)
            self.drain_started.set()
            await allow_drain.wait()
            self.drained = True

    def make_server(config):
        server = SlowDrainServer(config)
        servers.append(server)
        return server

    serve_task = asyncio.create_task(
        _serve_apps(
            object(),
            object(),
            listener_factory=lambda host, port: listeners[port - 8091],
            server_factory=make_server,
            signal_context_factory=signals.manage,
        )
    )
    while len(servers) < 2:
        await asyncio.sleep(0)
    await asyncio.gather(*(server.start_event.wait() for server in servers))
    assert signals.trigger is not None
    signals.trigger()
    await asyncio.gather(*(server.drain_started.wait() for server in servers))

    serve_task.cancel()
    serve_task.cancel()
    allow_drain.set()
    with pytest.raises(asyncio.CancelledError):
        await serve_task

    assert all(server.drained for server in servers)
    assert all(listener.closed for listener in listeners)
    assert signals.exited
