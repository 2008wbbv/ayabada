"""Ayabada: a self-hosting DevOps/SRE agent.

One system, two scopes:

- ``ayabada.brain`` + ``ayabada.bench``  — the *research* scope. A
  confidence-aware stopping/escalation wrapper around a standard tool-use
  loop, scored on ITBench-AA against a baseline that shares the same model
  and the same tools. Nothing outside this scope may influence a scored run.
- ``ayabada.heartbeat`` + ``ayabada.shell`` — the *product* scope. Seasonal
  anomaly detection that wakes the brain on real incidents, plus the
  self-host packaging, CLI and doc-handoff. Runs in production only.

The two scopes meet at exactly one interface: :class:`ayabada.snapshot.IncidentSnapshot`.
"""

__version__ = "0.1.0"
