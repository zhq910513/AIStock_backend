from __future__ import annotations

from dataclasses import dataclass
from sqlalchemy.orm import Session

from app.database.repo import Repo
from app.utils.time import now_shanghai


@dataclass
class ContraInputs:
    symbol: str
    data_window_start: object  # datetime
    data_window_end: object    # datetime
    old_contract_hash: str
    new_contract_hash: str
    old_decision_bundle_ref: str
    new_decision_bundle_ref: str
    old_decision: str
    new_decision: str
    confidence_delta: float
    position_delta: float
    guard_level_delta: int
    data_quality_delta: dict


class DifferentialAuditEngine:
    def write_contra(self, s: Session, inp: ContraInputs, severity: str) -> None:
        Repo(s).system_events.write_event(
            event_type="CONTRA_DECISION_EVENT",
            correlation_id=None,
            severity=severity,
            symbol=inp.symbol,
            payload={
                "event_type": "CONTRA_DECISION_EVENT",
                "symbol": inp.symbol,
                "data_window": {"start_ts": inp.data_window_start.isoformat(), "end_ts": inp.data_window_end.isoformat()},
                "old_contract_hash": inp.old_contract_hash,
                "new_contract_hash": inp.new_contract_hash,
                "old_decision_bundle_ref": inp.old_decision_bundle_ref,
                "new_decision_bundle_ref": inp.new_decision_bundle_ref,
                "diff_summary": {
                    "decision_direction_diff": f"{inp.old_decision}->{inp.new_decision}",
                    "confidence_delta": float(inp.confidence_delta),
                    "position_delta": float(inp.position_delta),
                    "guard_level_delta": int(inp.guard_level_delta),
                    "data_quality_delta": inp.data_quality_delta,
                },
                "time": now_shanghai().isoformat(),
            },
        )
