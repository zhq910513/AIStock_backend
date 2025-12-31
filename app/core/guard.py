from __future__ import annotations

from dataclasses import dataclass
from sqlalchemy.orm import Session

from app.database.repo import Repo


@dataclass
class GuardResult:
    guard_level: int  # 0/1/2/3
    veto: bool
    veto_code: str
    market_score: float


class Guard:
    """
    风控宪法：独立、具一票否决权。
    """
    def evaluate(self, s: Session, symbol: str, side: str, intended_qty: int) -> GuardResult:
        repo = Repo(s)
        st = repo.system_status.get_for_update()

        if st.panic_halt:
            return GuardResult(guard_level=3, veto=True, veto_code="PANIC_HALT", market_score=0.0)

        # Minimal baseline rules (extend by RuleSet, Position, Drawdown, Slippage ladder, etc.)
        if intended_qty <= 0:
            return GuardResult(guard_level=1, veto=True, veto_code="INVALID_QTY", market_score=0.0)

        return GuardResult(guard_level=0, veto=False, veto_code="", market_score=1.0)

    def enter_orange_on_orphan_fill(self, s: Session, veto_code: str = "ORPHAN_FILL") -> None:
        repo = Repo(s)
        repo.system_status.set_guard_level(level=2, veto=True, veto_code=veto_code)
