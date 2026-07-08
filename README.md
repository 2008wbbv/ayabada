# Ayabada — a self-hosting DevOps/SRE agent

One system, two lives, kept separate in the code:

- **The product**: a self-hosted agent that watches live metrics, wakes on
  real incidents, diagnoses, and hands a human a document.
- **The research paper**: a controlled experiment on one component — a
  **confidence-aware stopping/escalation mechanism** — evaluated on
  [ITBench-AA](https://huggingface.co/datasets/ArtificialAnalysis/ITBench-AA)
  against a baseline that shares the same base model and the same tools.

**Claim under test:** a confidence-aware agent that regulates when it stops
and escalates beats vanilla frontier models on an open SRE benchmark,
specifically by not over-investigating. ITBench-AA scores root-cause
identification with recall-gated precision — naming symptom entities beyond
the true root cause loses points, and long trajectories score worse, not
better. The measured weakness is not knowing when to stop.

## The one rule that can't break

In any scored run, the base model and tool access are **identical** between
arms; only the architecture (the confidence/stopping mechanism) varies. This
is enforced in code, not by convention: `ayabada.bench.parity` hashes each
arm's model, model parameters, tool schemas and task instructions into an
*environment fingerprint*, the runner refuses to start when fingerprints
differ, and the shared fingerprint is written into every results file.
Both arms are also built from a single LLM factory and a single toolbox
class, so they cannot diverge by construction.

Everything a human touches — heartbeat, CLI, docker packaging, doc-handoff —
runs in production, outside the scored loop.

## Layout

| Path | Scope | What it is |
|---|---|---|
| `ayabada/brain/` | research | Stock tool-use loop (`loop.py` — the baseline), confidence checkpoints (`confidence.py`), stopping/escalation policy (`policy.py`), and the wrapper arm (`agent.py`) |
| `ayabada/bench/` | research | ITBench-AA adapter (`itbench.py`, `ground_truth.py`), recall-gated precision scoring (`scoring.py`), parity guardrails (`parity.py`), head-to-head runner (`runner.py`) |
| `ayabada/heartbeat/` | product | Seasonal baseline (day-of-week × 15-min buckets, median+MAD), deseasonalized corroboration detector, persistence/hysteresis/cooldown wake gate, special-day calendar |
| `ayabada/shell/` | product | CLI, doc-handoff generator, docker packaging |
| `ayabada/snapshot.py` | shared | `IncidentSnapshot` — the one interface both scopes feed. The brain can't tell whether the benchmark or the heartbeat woke it |
| `ayabada/llm/` | shared | Anthropic client (stock model, manual loop) + deterministic scripted mock |

## Quick start

```bash
pip install -e ".[llm,dev]"
pytest                                   # 70 tests, all offline

# End-to-end product demo, no API key: synthetic incident → heartbeat wake
# → confidence brain → markdown handoff
ayabada demo

# Web dashboard (zero dependencies): live gate state, z-score chart with
# thresholds and wake markers, per-metric small multiples, self-host
# services panel, incident table with handoff viewer. Demo feed is
# time-compressed (1 tick = 15 min).
ayabada dashboard                        # http://127.0.0.1:8787/
ayabada dashboard --csv metrics.csv      # replay real metrics instead
ayabada dashboard --services services.yml  # health-check your own services

# Scored benchmark, offline plumbing check (scripted model)
ayabada bench fetch --limit 1
ayabada bench run --limit 1 --dry-run

# Real scored run (both arms, same model, same tools)
export ANTHROPIC_API_KEY=...
ayabada bench run --limit 5 --model claude-opus-4-8 --out results/

# Replay production metrics through the wake gate
ayabada heartbeat replay metrics.csv     # wide CSV: timestamp, metric columns
```

Or self-hosted: `docker compose run --rm agent demo`.

## The mechanism (brain)

The baseline arm is the vanilla loop: model + investigation tools, run until
it calls `submit_diagnosis` or hits the turn cap. The confidence arm runs the
*same* loop with two additions:

1. **Checkpoints** — every N turns the same stock model is asked for its
   current hypothesis and a calibrated probability that the hypothesis names
   exactly the true root-cause entities.
2. **Policy** — stop and submit when confidence crosses `stop_threshold`;
   escalate to a human when confidence stalls below `escalate_threshold` for
   `patience` checkpoints; at the turn budget, submit the best hypothesis if
   it clears the bar, otherwise escalate honestly instead of guessing.

Checkpoint calls are counted in the trajectory length, so reported wins pay
for the mechanism's own overhead. Escalations score 0 (tracked separately as
escalation rate) — an honest "I don't know" is the designed alternative to a
confidently wrong answer.

## The heartbeat (product)

Seasonal anomaly detection, no LLM involved: bucket history by
(day-of-week, 15-minute slot) with a cold-start fallback hierarchy; convert
every metric to a robust z-score against its own bucket (raw values are
never thresholded); require 2+ corroborating metrics; wake only after the
anomaly persists N intervals with at least one severe deviation; hysteresis
holds a streak through borderline intervals; a cooldown stops re-waking on
the same episode; flagged special days (Black Friday) widen the bands. In
the included simulation it produces zero false wakes over three clean weeks
and exactly one wake per injected incident.

## Watched services

The dashboard's services panel health-checks whatever you self-host,
declared in a YAML file:

```yaml
services:
  - name: website          # HTTP 2xx/3xx within timeout
    kind: http
    target: https://example.com/health
    interval: 30           # seconds (default 30)
  - name: postgres         # TCP connect
    kind: tcp
    target: "127.0.0.1:5432"
  - name: grafana          # docker container running
    kind: docker
    target: grafana
  - name: backup-timer     # exit code 0
    kind: command
    target: "systemctl is-active backups.timer"
```

Each service shows status (up / slow / failing / down), current latency with
a sparkline, uptime % over the recent window, and every up↔down transition
lands in an event log. `down` requires `down_after` consecutive failures
(default 2) so one dropped packet doesn't page anyone. Without `--services`
the demo watches its own HTTP/TCP endpoints plus a deliberately unreachable
example so all states are visible.

## Results format

`bench run` writes `records.jsonl` (one record per task × arm: predictions,
per-task precision/recall/score, trajectory length, confidence trace) and
`summary.json` (per-arm aggregates + the shared environment fingerprint).
The ablation — mechanism on vs off — is the diff of two arm summaries.

## Roadmap

- Reproduce the published leaderboard baseline with the official ITBench
  harness before claiming deltas (build order step 1; requires API budget).
- Memory across incidents, tested on held-out faults.
- Verify-before-act on AIOpsLab (live mitigation) if the mechanism needs to
  act rather than diagnose.

## Data

`data/itbench/data.jsonl` is the public SRE manifest of
[ArtificialAnalysis/ITBench-AA](https://huggingface.co/datasets/ArtificialAnalysis/ITBench-AA)
(CC-BY-4.0). Scenario telemetry is fetched on demand to
`~/.cache/ayabada/itbench/`. If you use this, cite IBM's ITBench paper
alongside the Artificial Analysis release.
