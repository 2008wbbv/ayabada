# Ayabada development notes

## The one rule that can't break

In any **scored** benchmark run, the base model and tool access must be
identical between the confidence arm and the baseline. Only the architecture
(stopping/escalation mechanism) may vary. This is enforced by
`ayabada/bench/parity.py` — do not weaken it, route around it, or add
arm-specific tools/context/prompts to `ayabada/bench/` or `ayabada/brain/`
without extending the parity fingerprint to cover the new surface.

Forbidden inside scored runs (fine in the product):
- web/StackOverflow browsing for extra context (unless given to both arms)
- model switching or handoff
- any tool one arm has that the other doesn't

## Two scopes, one interface

- Research scope: `ayabada/brain/` + `ayabada/bench/`. Deterministic,
  offline, parity-checked.
- Product scope: `ayabada/heartbeat/` + `ayabada/shell/`. Never imported by
  the research scope.
- They meet only at `ayabada/snapshot.py::IncidentSnapshot`. Keep it that way:
  a change that makes the brain aware of provenance beyond logging is a bug.

## Conventions

- Python ≥ 3.11, stdlib-first; `pyyaml` is the only hard dependency,
  `anthropic` is the `[llm]` extra. The heartbeat must stay stdlib-only.
- Tests are fully offline (scripted LLM, synthetic metrics, tmp fixtures).
  `pytest -q` must pass without network or an API key.
- Baseline arm behavior lives in `brain/loop.py`; the confidence arm
  (`brain/agent.py`) must reuse `execute_turn`/`initial_messages`/
  `SYSTEM_PROMPT` from it rather than duplicating loop logic — shared
  machinery is what makes the comparison clean.
- Checkpoint LLM calls count in `num_turns`. Don't "optimize" that away;
  honest trajectory accounting is part of the claim.
- ITBench telemetry cache: `~/.cache/ayabada/itbench/`. Never commit fetched
  scenario data; only `data/itbench/data.jsonl` (the manifest) is vendored.

## Commands

```bash
pip install -e ".[llm,dev]"
pytest -q                       # full offline suite
ayabada demo                    # product loop end-to-end, no API key
ayabada bench run --limit 1 --dry-run   # scored-path plumbing check
```
