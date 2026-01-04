from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.engine import SessionLocal
from app.database.repo import Repo
from app.database import models
from app.utils.time import now_shanghai_str
from app.utils.crypto import sha256_hex
from app.config import settings


router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"ok": True, "time": now_shanghai_str()}


@router.get("/status")
def status() -> dict:
    with SessionLocal() as s:
        repo = Repo(s)
        st = repo.system_status.get_for_update()
        s.commit()
        return {
            "time": now_shanghai_str(),
            "panic_halt": bool(st.panic_halt),
            "guard_level": int(st.guard_level),
            "veto": bool(st.veto),
            "veto_code": st.veto_code,
            "last_self_check_report_hash": st.last_self_check_report_hash,
            "last_self_check_time": st.last_self_check_time.isoformat() if st.last_self_check_time else None,
        }


@router.post("/admin/self_check")
def run_self_check() -> dict:
    """
    Minimal self-check report, required by trade gate in this skeleton.
    """
    with SessionLocal() as s:
        repo = Repo(s)
        st = repo.system_status.get_for_update()

        report = {
            "time": now_shanghai_str(),
            "versions": {
                "RuleSetVersionHash": settings.RULESET_VERSION_HASH,
                "StrategyContractHash": settings.STRATEGY_CONTRACT_HASH,
                "ModelSnapshotUUID": settings.MODEL_SNAPSHOT_UUID,
                "CostModelVersion": settings.COST_MODEL_VERSION,
                "CanonicalizationVersion": settings.CANONICALIZATION_VERSION,
                "FeatureExtractorVersion": settings.FEATURE_EXTRACTOR_VERSION,
            },
            "note": "minimal self-check: schema+versions+connectivity assumed OK",
        }
        report_hash = sha256_hex(str(report).encode("utf-8"))
        repo.system_status.set_self_check(report_hash)
        repo.system_events.write_event(
            event_type="SELF_CHECK_REPORT",
            correlation_id=None,
            severity="INFO",
            payload={"report": report, "report_hash": report_hash},
        )
        s.commit()
        return {"ok": True, "report_hash": report_hash, "report": report}


@router.get("/decisions")
def list_decisions(limit: int = 50) -> list[dict]:
    with SessionLocal() as s:
        rows = (
            s.execute(select(models.DecisionBundle).order_by(models.DecisionBundle.created_at.desc()).limit(limit))
            .scalars()
            .all()
        )
        return [
            {
                "decision_id": r.decision_id,
                "cid": r.cid,
                "account_id": r.account_id,
                "symbol": r.symbol,
                "decision": r.decision,
                "confidence": float((r.params or {}).get("confidence", 0.0)) if isinstance(r.params, dict) else None,
                "reason_code": r.reason_code,
                "created_at": r.created_at.isoformat(),
                "request_ids": r.request_ids,
                "feature_hash": r.feature_hash,
                "model_hash": r.model_hash,
                "data_quality": r.data_quality,
            }
            for r in rows
        ]


@router.get("/data_requests")
def list_data_requests(status: str | None = None, limit: int = 50) -> list[dict]:
    with SessionLocal() as s:
        q = select(models.DataRequest).order_by(models.DataRequest.created_at.desc())
        if status:
            q = q.where(models.DataRequest.status == status)
        rows = s.execute(q.limit(limit)).scalars().all()
        return [
            {
                "request_id": r.request_id,
                "dedupe_key": r.dedupe_key,
                "correlation_id": r.correlation_id,
                "account_id": r.account_id,
                "symbol": r.symbol,
                "purpose": r.purpose,
                "provider": r.provider,
                "endpoint": r.endpoint,
                "status": r.status,
                "attempts": r.attempts,
                "created_at": r.created_at.isoformat(),
                "sent_at": r.sent_at.isoformat() if r.sent_at else None,
                "deadline_at": r.deadline_at.isoformat() if r.deadline_at else None,
                "response_id": r.response_id,
                "last_error": r.last_error,
            }
            for r in rows
        ]


@router.get("/data_responses/{response_id}")
def get_data_response(response_id: str) -> dict:
    with SessionLocal() as s:
        r = s.get(models.DataResponse, response_id)
        if r is None:
            raise HTTPException(status_code=404, detail="not_found")
        return {
            "response_id": r.response_id,
            "request_id": r.request_id,
            "provider": r.provider,
            "endpoint": r.endpoint,
            "http_status": r.http_status,
            "errorcode": r.errorcode,
            "errmsg": r.errmsg,
            "quota_context": r.quota_context,
            "payload_sha256": r.payload_sha256,
            "received_at": r.received_at.isoformat(),
            "raw": r.raw,
        }


@router.get("/accounts")
def list_accounts() -> list[dict]:
    with SessionLocal() as s:
        repo = Repo(s)
        repo.accounts.ensure_accounts_seeded()
        rows = repo.accounts.list_accounts()
        s.commit()
        return [{"account_id": r.account_id, "broker_type": r.broker_type, "created_at": r.created_at.isoformat()} for r in rows]
