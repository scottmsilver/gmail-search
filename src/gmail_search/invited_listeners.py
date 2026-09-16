"""Run the invited browser and bearer gateway on private loopback listeners."""

from __future__ import annotations

import asyncio
import signal
import socket
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, ContextManager

import uvicorn


_LOOPBACK_HOST = "127.0.0.1"
_BROWSER_PORT = 8091
_GATEWAY_PORT = 8092


def _build_server_config(app: Any, *, port: int) -> uvicorn.Config:
    return uvicorn.Config(
        app,
        host=_LOOPBACK_HOST,
        port=port,
        access_log=False,
        log_config=None,
        # Uvicorn logs lifespan exception values and tracebacks verbatim. The
        # caller reports startup failure generically after supervised cleanup.
        log_level="critical",
        loop="asyncio",
        proxy_headers=False,
        reload=False,
        workers=1,
    )


def _open_listener(host: str, port: int) -> socket.socket:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen()
        listener.setblocking(False)
    except BaseException:
        listener.close()
        raise
    return listener


async def _run_server(server: Any, listener: Any) -> BaseException | None:
    """Contain fatal startup errors until the sibling has shut down."""
    try:
        await server.serve(sockets=[listener])
    except BaseException as exc:
        return exc
    return None


async def _wait_until_started(server: Any) -> None:
    while not server.started:
        await asyncio.sleep(0.01)


async def _drain_servers(
    servers: list[Any],
    server_tasks: list[asyncio.Task[BaseException | None]],
    auxiliary_tasks: list[asyncio.Task[Any]],
) -> list[BaseException | None]:
    for server in servers:
        server.should_exit = True
    for task in auxiliary_tasks:
        task.cancel()
    results = await asyncio.gather(*server_tasks, return_exceptions=True)
    await asyncio.gather(*auxiliary_tasks, return_exceptions=True)
    return results


class _ListenerServer(uvicorn.Server):
    """A uvicorn server whose signals are owned by the pair supervisor."""

    @contextmanager
    def capture_signals(self):
        yield


@contextmanager
def _signal_owner(trigger_shutdown: Callable[[], None]):
    loop = asyncio.get_running_loop()
    handled = (signal.SIGINT, signal.SIGTERM)
    previous = {signum: signal.getsignal(signum) for signum in handled}
    installed: list[signal.Signals] = []
    try:
        for signum in handled:
            loop.add_signal_handler(signum, trigger_shutdown)
            installed.append(signum)
        yield
    finally:
        for signum in installed:
            loop.remove_signal_handler(signum)
            signal.signal(signum, previous[signum])


async def _serve_apps(
    browser_app: Any,
    gateway_app: Any,
    *,
    listener_factory: Callable[[str, int], Any] = _open_listener,
    server_factory: Callable[[uvicorn.Config], Any] = _ListenerServer,
    signal_context_factory: Callable[
        [Callable[[], None]], ContextManager[None]
    ] = _signal_owner,
) -> None:
    listeners: list[Any] = []
    try:
        listeners.append(listener_factory(_LOOPBACK_HOST, _BROWSER_PORT))
        listeners.append(listener_factory(_LOOPBACK_HOST, _GATEWAY_PORT))
        servers = [
            server_factory(_build_server_config(browser_app, port=_BROWSER_PORT)),
            server_factory(_build_server_config(gateway_app, port=_GATEWAY_PORT)),
        ]
        shutdown_requested = asyncio.Event()
        with signal_context_factory(shutdown_requested.set):
            signal_task = asyncio.create_task(shutdown_requested.wait())
            gateway_task = asyncio.create_task(_run_server(servers[1], listeners[1]))
            gateway_ready_task = asyncio.create_task(_wait_until_started(servers[1]))
            server_tasks = [gateway_task]
            active_servers = [("gateway", servers[1])]
            pending_error: BaseException | None = None
            try:
                first_done, _ = await asyncio.wait(
                    [gateway_task, gateway_ready_task, signal_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                gateway_ready = (
                    gateway_ready_task in first_done
                    and gateway_task not in first_done
                    and signal_task not in first_done
                    and not gateway_task.done()
                    and not signal_task.done()
                )
                if gateway_ready:
                    browser_task = asyncio.create_task(
                        _run_server(servers[0], listeners[0])
                    )
                    server_tasks.append(browser_task)
                    active_servers.append(("browser", servers[0]))
                    await asyncio.wait(
                        [gateway_task, browser_task, signal_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
            except BaseException as exc:
                pending_error = exc
            drain_task = asyncio.create_task(
                _drain_servers(
                    servers,
                    server_tasks,
                    [signal_task, gateway_ready_task],
                )
            )
            while not drain_task.done():
                try:
                    await asyncio.shield(drain_task)
                except asyncio.CancelledError as exc:
                    if pending_error is None:
                        pending_error = exc
            results = drain_task.result()
            if pending_error is not None:
                raise pending_error
            for (name, _server), result in zip(
                active_servers, results, strict=True
            ):
                if isinstance(result, asyncio.CancelledError):
                    raise result
                if isinstance(result, SystemExit):
                    code = result.code if isinstance(result.code, int) else 1
                    raise SystemExit(code) from None
                if isinstance(result, BaseException):
                    raise RuntimeError(f"{name} listener failed") from None
            for name, server in active_servers:
                if not server.started:
                    raise RuntimeError(f"{name} listener failed to start")
    finally:
        for listener in listeners:
            listener.close()


async def serve_apps(browser_app: Any, gateway_app: Any) -> None:
    """Serve both invited surfaces until one exits or a process signal arrives."""
    await _serve_apps(browser_app, gateway_app)
