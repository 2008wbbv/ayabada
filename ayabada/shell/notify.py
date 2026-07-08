"""Webhook notifications: the agent tells someone what it did.

One :class:`Notifier` fans a notification out to every configured URL in
one of three wire formats:

- ``json``  — generic ``POST`` with a JSON body (title, body, priority, extras)
- ``ntfy``  — plain-text body with ``Title``/``Priority`` headers (ntfy.sh)
- ``slack`` — ``{"text": "*title*\\nbody"}`` (Slack/Mattermost-compatible)

Sending never raises and never blocks the caller for more than the timeout;
failures are counted and visible on the services panel via ``stats()``.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

FORMATS = ("json", "ntfy", "slack")


@dataclass
class Notifier:
    urls: list[str]
    fmt: str = "json"
    timeout: float = 5.0
    log: Callable[[str], None] = print
    sent: int = field(default=0, init=False)
    failed: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        if self.fmt not in FORMATS:
            raise ValueError(f"unknown notify format {self.fmt!r} (expected one of {FORMATS})")

    def send(
        self,
        title: str,
        body: str,
        priority: str = "default",
        extra: dict[str, Any] | None = None,
    ) -> int:
        """Deliver to every URL; returns how many deliveries succeeded."""
        ok = 0
        for url in self.urls:
            request = self._build(url, title, body, priority, extra or {})
            try:
                with urllib.request.urlopen(request, timeout=self.timeout):  # noqa: S310
                    ok += 1
            except Exception as exc:  # noqa: BLE001 — notification failure must not propagate
                self.log(f"notify: {url}: {type(exc).__name__}: {exc}"[:200])
        with self._lock:
            self.sent += ok
            self.failed += len(self.urls) - ok
        return ok

    def _build(
        self, url: str, title: str, body: str, priority: str, extra: dict[str, Any]
    ) -> urllib.request.Request:
        if self.fmt == "ntfy":
            return urllib.request.Request(
                url,
                data=body.encode(),
                headers={"Title": title, "Priority": priority},
                method="POST",
            )
        if self.fmt == "slack":
            payload = {"text": f"*{title}*\n{body}"}
        else:
            payload = {"title": title, "body": body, "priority": priority, **extra}
        return urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"sent": self.sent, "failed": self.failed}
