from __future__ import annotations

import secrets
import hashlib
from fastapi import APIRouter, HTTPException
from sqlalchemy import select, desc

from app.database.engine import SessionLocal
from app.database import models
from app.database.repo import Repo
from app.utils.time import now_shanghai, now_shanghai_str, fmt_ts_millis
from app.utils.crypto import verify_ed25519_signature
from app.config import settings
from app.core.reconciler import Reconciler


router = APIRouter()


@router.get("/health")
def health():
    return {"ok": True, "time": now_shanghai_str()}


@router.get("/system/status")
def system_status():
    with SessionLocal() as s:
        st = s.execute(select(models.SystemStatus).where(models.SystemStatus.id == 1)).scalar_one_or_none()
        if st is None:
            st = models.SystemStatus(id=1, updated_at=now_shanghai())
            s.add(st)
            s.commit()
        return {
            "guard_level": st.guard_level,
            "veto": st.veto,
            "veto_code": st.veto_code,
            "panic_halt": st.panic_halt,
            "challenge_code": st.challenge_code,
            "last_self_check_report_hash": st.last_self_check_report_hash,
            "last_self_check_time": fmt_ts_millis(st.last_self_check_time) if st.last_self_check_time else None,
            "updated_at": fmt_ts_millis(st.updated_at),
        }


@router.post("/system/panic_halt")
def panic_halt(veto_code: str = "MANUAL_PANIC_HALT"):
    with SessionLocal() as s:
        repo = Repo(s)
        repo.system_status.set_panic_halt(veto_code=veto_code)
        repo.system_events.write_event(
            event_type="PANIC_HALT",
            correlation_id=None,
            severity="CRITICAL",
            payload={"veto_code": veto_code, "time": now_shanghai_str()},
        )
        s.commit()
    return {"ok": True, "panic_halt": True}


@router.post("/system/reset/request")
def reset_request(
    devops_key_id: str | None = None,
    signature_b64: str | None = None,
    request_nonce: str | None = None,
):
    if settings.RESET_REQUIRE_DEVOPS_SIGNATURE:
        if not (devops_key_id and signature_b64 and request_nonce):
            raise HTTPException(status_code=400, detail="devops signature required")

        message = f"RESET_REQUEST|{request_nonce}".encode("utf-8")

        with SessionLocal() as s:
            key = s.get(models.GuardianKey, devops_key_id)
            if key is None or key.revoked or key.role != "DEVOPS":
                raise HTTPException(status_code=403, detail="Invalid devops key")

            ok = verify_ed25519_signature(key.public_key_b64, message, signature_b64)
            if not ok:
                raise HTTPException(status_code=403, detail="DevOps signature verification failed")

    challenge = secrets.token_urlsafe(24)
    with SessionLocal() as s:
        repo = Repo(s)
        repo.system_status.set_challenge(challenge)
        repo.system_events.write_event(
            event_type="RESET_REQUESTED",
            correlation_id=None,
            severity="WARN",
            payload={
                "challenge_code": challenge,
                "devops_key_id": devops_key_id,
                "request_nonce": request_nonce,
                "time": now_shanghai_str(),
            },
        )
        s.commit()
    return {"challenge_code": challenge}


@router.post("/system/reset/confirm")
def reset_confirm(key_id: str, signature_b64: str):
    with SessionLocal() as s:
        repo = Repo(s)
        st = repo.system_status.get_for_update()
        if not st.challenge_code:
            raise HTTPException(status_code=400, detail="No challenge_code set")
        challenge = st.challenge_code

        key = s.get(models.GuardianKey, key_id)
        if key is None or key.revoked or key.role != "RISK_OFFICER":
            raise HTTPException(status_code=403, detail="Invalid risk officer key")

        ok = verify_ed25519_signature(key.public_key_b64, challenge.encode("utf-8"), signature_b64)
        if not ok:
            raise HTTPException(status_code=403, detail="Signature verification failed")

        report = {
            "db_ok": True,
            "challenge_code": challenge,
            "time": now_shanghai_str(),
            "note": "Minimal self-check. Extend: latency, balances, event-chain integrity.",
        }
        report_hash = hashlib.sha256(str(report).encode("utf-8")).hexdigest()

        repo.system_events.write_event(
            event_type="SELF_CHECK_REPORT",
            correlation_id=None,
            severity="INFO",
            payload={"report": report, "report_hash": report_hash},
        )

        repo.system_status.set_self_check(report_hash)

        repo.system_status.reset_from_panic()
        repo.system_events.write_event(
            event_type="RESET_COMPLETED",
            correlation_id=None,
            severity="WARN",
            payload={"report_hash": report_hash, "time": now_shanghai_str()},
        )
        s.commit()

    return {"ok": True, "report_hash": report_hash}


@router.post("/reconcile/decision")
def reconcile_decision(
    snapshot_id: str,
    decided_cid: str,
    decided_broker_order_id: str | None,
    signer_key_id: str,
    signature_b64: str,
):
    """
    Signed reconcile decision:
    message = "RECONCILE_DECISION|<snapshot_id>|<decided_cid>|<decided_broker_order_id_or_empty>"
    """
    with SessionLocal() as s:
        rec = Reconciler()
        try:
            decision_id = rec.apply_reconcile_decision_signed(
                s=s,
                snapshot_id=snapshot_id,
                decided_cid=decided_cid,
                decided_broker_order_id=decided_broker_order_id,
                signer_key_id=signer_key_id,
                signature_b64=signature_b64,
            )
            s.commit()
            return {"ok": True, "decision_id": decision_id}
        except Exception as e:
            s.rollback()
            raise HTTPException(status_code=400, detail=str(e))


@router.get("/events")
def list_events(limit: int = 50):
    limit = max(1, min(200, limit))
    with SessionLocal() as s:
        rows = s.execute(select(models.SystemEvent).order_by(desc(models.SystemEvent.time)).limit(limit)).scalars().all()
        return [
            {
                "id": r.id,
                "event_type": r.event_type,
                "severity": r.severity,
                "symbol": r.symbol,
                "correlation_id": r.correlation_id,
                "payload": r.payload,
                "time": fmt_ts_millis(r.time),
            }
            for r in rows
        ]


@router.get("/orders")
def list_orders(limit: int = 50):
    limit = max(1, min(200, limit))
    with SessionLocal() as s:
        rows = s.execute(select(models.Order).limit(limit)).scalars().all()
        return [
            {
                "cid": o.cid,
                "symbol": o.symbol,
                "side": o.side,
                "state": o.state,
                "client_order_id": o.client_order_id,
                "broker_order_id": o.broker_order_id,
                "qty_int": o.qty_int,
                "limit_price_int64": o.limit_price_int64,
                "metadata_hash": o.metadata_hash,
                "version_id": o.version_id,
                "last_transition_id": o.last_transition_id,
                "updated_at": fmt_ts_millis(o.updated_at),
            }
            for o in rows
        ]


@router.get("/order_anchors")
def list_order_anchors(limit: int = 50):
    limit = max(1, min(200, limit))
    with SessionLocal() as s:
        rows = s.execute(select(models.OrderAnchor).limit(limit)).scalars().all()
        return [
            {
                "cid": r.cid,
                "client_order_id": r.client_order_id,
                "broker_order_id": r.broker_order_id,
                "request_uuid": r.request_uuid,
                "ack_hash": r.ack_hash,
                "raw_request_hash": r.raw_request_hash,
                "raw_response_hash": r.raw_response_hash,
                "created_at": fmt_ts_millis(r.created_at),
            }
            for r in rows
        ]
