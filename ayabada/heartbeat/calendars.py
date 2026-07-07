"""Special-day flagging so Black Friday doesn't wake the agent every year.

A :class:`SpecialDayCalendar` is just a set of dates the operator expects to
be off-pattern (holidays, sale days, planned load tests). On those days the
detector widens its bands by ``band_multiplier`` instead of trusting the
seasonal baseline, which was learned from ordinary days.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class SpecialDayCalendar:
    dates: set[date] = field(default_factory=set)
    band_multiplier: float = 2.0

    def add(self, day: date) -> None:
        self.dates.add(day)

    def multiplier_for(self, ts: datetime) -> float:
        return self.band_multiplier if ts.date() in self.dates else 1.0
