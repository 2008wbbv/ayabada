"""The ayabada CLI — the product's terminal surface.

Subcommands:

- ``ayabada bench fetch``    download ITBench-AA scenario snapshots
- ``ayabada bench run``      scored head-to-head (baseline vs confidence arm)
- ``ayabada heartbeat replay``  run the wake trigger over a metrics CSV
- ``ayabada demo``           end-to-end product demo: synthetic incident →
                             heartbeat wake → brain → doc-handoff (offline,
                             scripted model, no API key needed)

Scored runs (``bench run``) go through the parity-checked runner; everything
else is product scope.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

from ayabada.bench.itbench import fetch_scenario, load_manifest
from ayabada.bench.runner import DEFAULT_ARMS, run_head_to_head


def _select_scenarios(names: list[str] | None, limit: int | None):
    scenarios = load_manifest()
    if names:
        wanted = set(names)
        scenarios = [s for s in scenarios if s.scenario_id in wanted or s.id_aa in wanted]
    if limit:
        scenarios = scenarios[:limit]
    return scenarios


def cmd_bench_fetch(args: argparse.Namespace) -> int:
    scenarios = _select_scenarios(args.scenario, args.limit)
    for scenario in scenarios:
        print(f"== {scenario.scenario_id}")
        fetch_scenario(scenario.scenario_id, include_metrics=args.metrics)
    return 0


def cmd_bench_run(args: argparse.Namespace) -> int:
    scenarios = _select_scenarios(args.scenario, args.limit)
    if not scenarios:
        print("no scenarios selected", file=sys.stderr)
        return 2

    if args.dry_run:
        from ayabada.llm.mock import ScriptedLLM

        def factory():
            return ScriptedLLM(
                script=[
                    {"text": "checking alerts", "tool": "get_alerts", "input": {}},
                    '{"confidence": 0.2, "hypothesis": [], "rationale": "just started"}',
                    {"text": "events", "tool": "get_events", "input": {"event_type": "Warning"}},
                    '{"confidence": 0.3, "hypothesis": [], "rationale": "inconclusive"}',
                ]
            )
    else:
        from ayabada.llm.anthropic_client import AnthropicClient

        def factory():
            return AnthropicClient(model=args.model)

    out_dir = Path(args.out)
    result = run_head_to_head(scenarios, factory, DEFAULT_ARMS, out_dir=out_dir)
    print()
    print(json.dumps(result.summary(), indent=2))
    print(f"\nrecords: {out_dir / 'records.jsonl'}")
    return 0


def cmd_heartbeat_replay(args: argparse.Namespace) -> int:
    from ayabada.heartbeat.monitor import Heartbeat, HeartbeatConfig

    hb = Heartbeat(HeartbeatConfig())
    wakes = 0
    with open(args.csv, newline="") as fh:
        reader = csv.DictReader(fh)
        ts_field = reader.fieldnames[0] if reader.fieldnames else "timestamp"
        for row in reader:
            ts = datetime.fromisoformat(row[ts_field])
            observations = {
                key: float(value)
                for key, value in row.items()
                if key != ts_field and value not in ("", None)
            }
            snapshot = hb.ingest(ts, observations)
            if snapshot is not None:
                wakes += 1
                print(f"WAKE {snapshot.id}: {snapshot.extra.get('gate_reason', '')}")
                if args.out:
                    path = Path(args.out) / f"{snapshot.id}.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({
                        "id": snapshot.id,
                        "description": snapshot.description,
                        "alerts": [a.to_dict() for a in snapshot.alerts],
                    }, indent=2))
    print(f"replay complete: {wakes} wake(s), final state {hb.state.value}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Synthetic incident → heartbeat wake → brain run → handoff doc."""
    from ayabada.brain.agent import run_confidence_loop
    from ayabada.brain.tools import SnapshotToolbox
    from ayabada.heartbeat.monitor import Heartbeat, HeartbeatConfig
    from ayabada.llm.mock import ScriptedLLM
    from ayabada.shell.handoff import render_handoff

    random.seed(11)
    hb = Heartbeat(HeartbeatConfig())
    ts = datetime(2026, 5, 4, 0, 0)

    def metrics(t: datetime) -> dict[str, float]:
        frac = (t.hour * 60 + t.minute) / 1440
        return {
            "traffic": max(1000 + 600 * math.sin(2 * math.pi * (frac - 0.3)) + random.gauss(0, 25), 50),
            "error_rate": max(random.gauss(0.01, 0.003), 0),
            "p95_latency_ms": max(random.gauss(180, 12), 1),
        }

    print("building 3 weeks of seasonal history…")
    for _ in range(3 * 7 * 96):
        hb.ingest(ts, metrics(ts))
        ts += timedelta(minutes=15)

    print("injecting incident (error rate x40, latency x4)…")
    snapshot = None
    for _ in range(12):
        obs = metrics(ts)
        obs["error_rate"] = 0.4 + random.gauss(0, 0.02)
        obs["p95_latency_ms"] = 800 + random.gauss(0, 40)
        snapshot = hb.ingest(ts, obs) or snapshot
        ts += timedelta(minutes=15)
        if snapshot:
            break
    if snapshot is None:
        print("heartbeat did not wake — unexpected", file=sys.stderr)
        return 1
    print(f"heartbeat woke: {snapshot.id}")

    llm = ScriptedLLM(
        script=[
            {"text": "inspecting alerts", "tool": "get_alerts", "input": {}},
            '{"confidence": 0.55, "hypothesis": [{"name": "checkout-service", "kind": "Service"}], '
            '"rationale": "error rate and latency both spiked; checkout is the common dependency"}',
            {"text": "confirming", "tool": "list_entities", "input": {}},
            '{"confidence": 0.85, "hypothesis": [{"name": "checkout-service", "kind": "Service"}], '
            '"rationale": "deviations corroborate a checkout-service regression"}',
        ]
    )
    run = run_confidence_loop(llm, SnapshotToolbox(snapshot))
    doc = render_handoff(snapshot, run)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    print(f"handoff written: {out}")
    print()
    print(doc)
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    from ayabada.shell.dashboard import run_dashboard

    run_dashboard(
        host=args.host,
        port=args.port,
        csv_path=args.csv,
        tick_seconds=args.tick,
        model=args.model,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ayabada", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("bench", help="ITBench-AA benchmark commands (scored scope)")
    bench_sub = bench.add_subparsers(dest="bench_command", required=True)

    fetch = bench_sub.add_parser("fetch", help="download scenario snapshots")
    fetch.add_argument("--scenario", action="append", help="scenario id (repeatable)")
    fetch.add_argument("--limit", type=int, help="first N scenarios")
    fetch.add_argument("--metrics", action="store_true", help="also fetch per-pod metric TSVs")
    fetch.set_defaults(func=cmd_bench_fetch)

    run = bench_sub.add_parser("run", help="head-to-head scored run")
    run.add_argument("--scenario", action="append", help="scenario id (repeatable)")
    run.add_argument("--limit", type=int, help="first N scenarios")
    run.add_argument("--model", default="claude-opus-4-8", help="base model (both arms)")
    run.add_argument("--out", default="results", help="output directory")
    run.add_argument("--dry-run", action="store_true", help="scripted model, no API key")
    run.set_defaults(func=cmd_bench_run)

    heartbeat = sub.add_parser("heartbeat", help="production wake trigger")
    hb_sub = heartbeat.add_subparsers(dest="heartbeat_command", required=True)
    replay = hb_sub.add_parser("replay", help="replay a metrics CSV through the gate")
    replay.add_argument("csv", help="wide CSV: timestamp column first, one column per metric")
    replay.add_argument("--out", help="directory to write wake snapshots into")
    replay.set_defaults(func=cmd_heartbeat_replay)

    demo = sub.add_parser("demo", help="offline end-to-end demo (no API key)")
    demo.add_argument("--out", default="handoff-demo.md", help="handoff output path")
    demo.set_defaults(func=cmd_demo)

    dashboard = sub.add_parser("dashboard", help="web dashboard (heartbeat + incidents)")
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8787)
    dashboard.add_argument("--csv", help="replay a metrics CSV instead of the synthetic demo feed")
    dashboard.add_argument(
        "--tick", type=float, default=1.0,
        help="seconds per 15-min interval in demo mode (time compression)",
    )
    dashboard.add_argument(
        "--model",
        help="run the real brain on wakes via the Anthropic API (default: scripted demo brain)",
    )
    dashboard.set_defaults(func=cmd_dashboard)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
