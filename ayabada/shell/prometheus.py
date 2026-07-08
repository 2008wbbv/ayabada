"""Prometheus ingestion: point the heartbeat at a real monitoring stack.

The operator maps heartbeat metric names to PromQL expressions; the feed
polls the instant-query API on the heartbeat's interval and hands each
round of observations to the same gate the demo and CSV feeds use:

.. code-block:: yaml

    prometheus:
      url: http://prometheus:9090
      interval: 900        # seconds per heartbeat interval (default 900 = 15 min)
    queries:
      traffic: sum(rate(http_requests_total[5m]))
      error_rate: >-
        sum(rate(http_requests_total{code=~"5.."}[5m]))
        / sum(rate(http_requests_total[5m]))
      p95_latency_ms: >-
        histogram_quantile(0.95,
          sum(rate(http_request_duration_seconds_bucket[5m])) by (le)) * 1000

Expressions should aggregate to a single value; a multi-element vector is
summed (with that noted in the log) so a forgotten ``sum()`` degrades
loudly rather than silently. Failed queries drop that metric for the
interval — corroboration still applies to whatever arrived, and an interval
with no successful queries is skipped entirely.
"""

from __future__ import annotations

import json
import threading
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import yaml

DEFAULT_INTERVAL = 900.0


@dataclass(frozen=True)
class PrometheusConfig:
    url: str
    queries: dict[str, str]
    interval: float = DEFAULT_INTERVAL
    timeout: float = 10.0


def load_prometheus_config(path: str) -> PrometheusConfig:
    with open(path) as fh:
        doc = yaml.safe_load(fh) or {}
    prom = doc.get("prometheus") or {}
    url = str(prom.get("url", "")).rstrip("/")
    queries = doc.get("queries") or {}
    if not url:
        raise ValueError(f"{path}: prometheus.url is required")
    if not isinstance(queries, dict) or not queries:
        raise ValueError(f"{path}: a non-empty 'queries' mapping is required")
    return PrometheusConfig(
        url=url,
        queries={str(k): str(v) for k, v in queries.items()},
        interval=float(prom.get("interval", DEFAULT_INTERVAL)),
        timeout=float(prom.get("timeout", 10.0)),
    )


def query_instant(base_url: str, expr: str, timeout: float = 10.0) -> float:
    """One instant query -> float. Raises on transport/shape errors."""
    url = f"{base_url}/api/v1/query?" + urllib.parse.urlencode({"query": expr})
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        payload = json.loads(resp.read())
    if payload.get("status") != "success":
        raise ValueError(f"prometheus error: {payload.get('error', 'unknown')}")
    data = payload.get("data") or {}
    result_type = data.get("resultType")
    result = data.get("result")
    if result_type == "scalar":
        return float(result[1])
    if result_type == "vector":
        if not result:
            raise ValueError("empty vector result")
        values = [float(item["value"][1]) for item in result]
        return sum(values)  # multi-element vectors are summed (see module doc)
    raise ValueError(f"unsupported resultType: {result_type}")


@dataclass
class PrometheusPoller:
    """One round of queries -> observations dict (partial on errors)."""

    config: PrometheusConfig
    log: Callable[[str], None] = print
    consecutive_failures: int = field(default=0, init=False)

    def poll(self) -> dict[str, float]:
        observations: dict[str, float] = {}
        errors: list[str] = []
        for metric, expr in self.config.queries.items():
            try:
                observations[metric] = query_instant(
                    self.config.url, expr, self.config.timeout
                )
            except Exception as exc:  # noqa: BLE001 — a bad query must not kill the feed
                errors.append(f"{metric}: {type(exc).__name__}: {exc}")
        if errors:
            self.log(f"prometheus: {len(errors)} quer{'y' if len(errors)==1 else 'ies'} failed — "
                     + "; ".join(errors)[:300])
        self.consecutive_failures = self.consecutive_failures + 1 if not observations else 0
        return observations


class PrometheusFeed(threading.Thread):
    """Poll Prometheus each interval and drive the heartbeat with the result."""

    def __init__(
        self,
        state: Any,  # DashboardState (kept untyped to avoid an import cycle)
        config: PrometheusConfig,
        brain_factory: Callable[[], Any],
        log: Callable[[str], None] = print,
    ) -> None:
        super().__init__(daemon=True, name="ayabada-prometheus-feed")
        self.state = state
        self.poller = PrometheusPoller(config, log=log)
        self.config = config
        self.brain_factory = brain_factory
        self.stop_event = threading.Event()

    def tick(self) -> None:
        """One poll + ingest; extracted so tests can drive it synchronously."""
        from ayabada.shell.dashboard import run_brain

        observations = self.poller.poll()
        self.state.feed_beat()
        if not observations:
            return
        snapshot = self.state.record_interval(datetime.now(), observations)
        if snapshot is not None:
            run_brain(self.state, snapshot, self.brain_factory)

    def run(self) -> None:
        while not self.stop_event.is_set():
            self.tick()
            self.stop_event.wait(self.config.interval)
