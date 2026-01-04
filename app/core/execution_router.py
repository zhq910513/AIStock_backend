from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.config import settings
from app.database.repo import Repo
from app.core.broker_adapter import BrokerAdapter, MockBrokerAdapter


@dataclass
class ExecutionRoute:
    account_id: str
    broker: BrokerAdapter


class ExecutionRouter:
    """
    Multi-account routing (minimal):
    - choose an account_id deterministically for a symbol
    - return a broker adapter bound to that account
    """
    def route(self, s: Session, symbol: str) -> ExecutionRoute:
        repo = Repo(s)
        repo.accounts.ensure_accounts_seeded()

        accounts = repo.accounts.list_accounts()
        if not accounts:
            account_id = settings.DEFAULT_ACCOUNT_ID
        else:
            idx = abs(hash(symbol)) % len(accounts)
            account_id = accounts[idx].account_id

        broker = MockBrokerAdapter(account_id=account_id)
        return ExecutionRoute(account_id=account_id, broker=broker)
