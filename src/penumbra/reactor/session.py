"""Reactor session plumbing, shared by every model adapter.

Reactor-specific command names never leave this package. Everything above it talks
to `Perturbation` and `Policy` interfaces.

The SDK is callback-driven off its own control thread; this wraps it in an asyncio
context manager that (a) waits properly for READY, (b) collects every model event so
an experiment record can carry the full server-side story, and (c) never lets a
credential reach a log line.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from reactor_sdk import Reactor, ReactorStatus
from reactor_sdk.errors import (
    DisconnectedError, InvalidStateError, NetworkError, RateLimitedError,
    RequestTimeoutError, ServerError, SessionTerminalError, TransportError,
)

from ..config import reactor_api_key

#: Failures of the connection rather than of the work. Retrying these is legitimate;
#: retrying a BadRequestError or an UnauthorizedError would just repeat a real mistake.
#: They are enumerated one by one rather than caught via their shared ReactorError base,
#: because that base also covers those genuine mistakes.
#:
#: Two of these earn their place by observation rather than by name:
#:
#: `InvalidStateError` - when a session leaves the ready state its publish does not
#: survive, and every subsequent push_frame raises this. A dropped connection wearing a
#: state error's clothes.
#:
#: `NetworkError` - and note what it is NOT. Every reactor_sdk error derives from
#: ReactorError, which derives from Exception, NOT from RuntimeError. Code that caught
#: `(TimeoutError, RuntimeError)` believing it was retrying transport failures was
#: catching none of them, for the whole life of the project. That is exactly how the
#: runner's rollout retry looked correct while being unable to catch the most common
#: failure it existed for.
TRANSPORT_ERRORS = (
    DisconnectedError, InvalidStateError, NetworkError, RateLimitedError,
    RequestTimeoutError, ServerError, SessionTerminalError, TransportError,
    ConnectionError, OSError, asyncio.TimeoutError,
)


log = logging.getLogger("penumbra.reactor")


def _status_value(status: Any) -> str:
    return getattr(status, "value", str(status))


@dataclass
class SessionLog:
    """Everything the server told us, kept for the experiment record."""

    model: str
    status_timeline: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    session_id: str | None = None
    connect_seconds: float | None = None

    def events_of(self, name: str) -> list[dict]:
        return [e for e in self.events if e.get("type") == name]


class ReactorSession:
    """One connected Reactor model.

    Usage::

        async with ReactorSession("xmax/x2") as s:
            await s.command("set_prompt", {"prompt": "..."})
    """

    #: No Reactor operation may be awaited forever. A wedged session must fail the
    #: experiment it belongs to, not hang the whole run: this was observed live, with
    #: a policy rollout stalling indefinitely after a successful connect because
    #: `send_command` and `publish_track` were awaited without a bound.
    DEFAULT_OP_TIMEOUT = 60.0

    def __init__(
        self,
        model: str,
        *,
        connect_timeout: float = 180.0,
        op_timeout: float = DEFAULT_OP_TIMEOUT,
    ) -> None:
        self.model = model
        self.connect_timeout = connect_timeout
        self.op_timeout = op_timeout
        self.log = SessionLog(model=model)
        self._reactor: Reactor | None = None
        self._ready: asyncio.Future | None = None
        self._event_hooks: list[Callable[[dict], None]] = []

    # -- lifecycle ---------------------------------------------------------

    @property
    def reactor(self) -> Reactor:
        if self._reactor is None:
            raise RuntimeError(f"{self.model}: session is not open")
        return self._reactor

    def on_event(self, hook: Callable[[dict], None]) -> None:
        """Called for every model event, on the event loop."""
        self._event_hooks.append(hook)

    async def __aenter__(self) -> ReactorSession:
        t0 = time.time()
        loop = asyncio.get_running_loop()
        self._ready = loop.create_future()
        r = Reactor(self.model, api_key=reactor_api_key())
        self._reactor = r

        @r.on_status
        def _on_status(status: Any) -> None:
            value = _status_value(status)
            self.log.status_timeline.append({"t": round(time.time() - t0, 3), "status": value})
            if value == ReactorStatus.READY.value and self._ready and not self._ready.done():
                self._ready.set_result(True)

        @r.on_message
        def _on_message(msg: Any) -> None:
            if isinstance(msg, dict):
                self.log.events.append(msg)
                for hook in self._event_hooks:
                    try:
                        hook(msg)
                    except Exception:  # noqa: BLE001
                        log.exception("event hook failed")

        @r.on_error
        def _on_error(err: Any) -> None:
            self.log.errors.append(f"{type(err).__name__}: {err}")

        await asyncio.wait_for(r.connect(), timeout=self.connect_timeout)
        await asyncio.wait_for(asyncio.shield(self._ready), timeout=self.connect_timeout)
        self.log.connect_seconds = round(time.time() - t0, 3)
        self.log.session_id = r.session_id
        log.info("%s ready in %.1fs", self.model, self.log.connect_seconds)
        return self

    async def __aexit__(self, *exc: object) -> None:
        r = self._reactor
        self._reactor = None
        if r is None:
            return
        try:
            await asyncio.wait_for(r.disconnect(), timeout=30)
        except Exception:  # noqa: BLE001
            log.debug("%s: disconnect did not complete cleanly", self.model)
        finally:
            r.close()

    # -- operations --------------------------------------------------------

    async def command(self, name: str, payload: dict | None = None) -> Any:
        try:
            return await asyncio.wait_for(
                self.reactor.send_command(name, payload or {}), timeout=self.op_timeout
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"{self.model}: command {name!r} did not complete within "
                f"{self.op_timeout}s. Session {self.log.session_id}"
            ) from exc

    async def publish(self, track: str):
        try:
            return await asyncio.wait_for(
                self.reactor.publish_track(track), timeout=self.op_timeout
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"{self.model}: publishing track {track!r} did not complete within "
                f"{self.op_timeout}s. Session {self.log.session_id}"
            ) from exc

    def track(self, name: str):
        return self.reactor.track(name)

    async def upload(self, path_or_bytes: Any, **kw: Any):
        try:
            return await asyncio.wait_for(
                self.reactor.upload_file(path_or_bytes, **kw), timeout=self.op_timeout
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"{self.model}: file upload did not complete within {self.op_timeout}s. "
                f"Session {self.log.session_id}"
            ) from exc
