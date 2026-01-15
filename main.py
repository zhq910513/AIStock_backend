import asyncio
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.router import router
from app.config import settings
from app.core.orchestrator import Orchestrator
from app.core.labeling_pipeline import LabelingPipeline
from app.database.engine import init_engine, init_schema_check, SessionLocal
from app.database.repo import Repo


def _now_shanghai_str() -> str:
    dt = datetime.now(tz=ZoneInfo("Asia/Shanghai"))
    # YYYY-MM-DD HH:MM:SS.mmm
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d}"


def _normalize_root_path(root_path: str | None) -> str:
    rp = (root_path or "").strip()
    if not rp:
        return ""
    if not rp.startswith("/"):
        rp = "/" + rp
    # remove trailing slash (except "/")
    if rp != "/" and rp.endswith("/"):
        rp = rp[:-1]
    return rp


app = FastAPI(
    title="AIStock_backend (QEE-S³/S³.1)",
    version="2.0.0",
    # Reverse-proxy aware Swagger/OpenAPI paths:
    root_path=settings.API_ROOT_PATH,
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
    swagger_ui_oauth2_redirect_url="/docs/oauth2-redirect",
)

app.include_router(router)

_orchestrator: Orchestrator | None = None
_orchestrator_task: asyncio.Task | None = None
_labeling_pipeline: LabelingPipeline | None = None


# ----------------------------
# Health endpoints
# ----------------------------
def _health_payload() -> dict:
    return {
        "ok": True,
        "service": "aistock-backend",
        "version": getattr(settings, "APP_VERSION", None) or os.getenv("APP_VERSION", app.version),
        "ts": _now_shanghai_str(),
        "api_root_path": _normalize_root_path(getattr(settings, "API_ROOT_PATH", "")),
    }


async def _health_handler():
    return JSONResponse(_health_payload())


# Always provide a plain health endpoint (useful for direct container checks)
app.add_api_route("/ui/health", _health_handler, methods=["GET"], tags=["ui"])

# Also provide a prefixed health endpoint to match reverse-proxy setups that do NOT rewrite paths
# e.g. external: /aistock/api/ui/health -> upstream receives the full prefixed path
_rp = _normalize_root_path(getattr(settings, "API_ROOT_PATH", ""))
if _rp and _rp != "/":
    prefixed_path = f"{_rp}/ui/health"
    if prefixed_path != "/ui/health":
        app.add_api_route(prefixed_path, _health_handler, methods=["GET"], tags=["ui"])

# Optional: legacy /health (your nginx has /aistock/health -> /health)
app.add_api_route("/health", _health_handler, methods=["GET"], tags=["ui"])


@app.on_event("startup")
async def _startup() -> None:
    global _orchestrator, _orchestrator_task, _labeling_pipeline

    init_engine()
    init_schema_check()

    # Ensure SystemStatus row exists (keeps /status deterministic and avoids first-hit races).
    with SessionLocal() as s:
        Repo(s).system_status.get_for_update()
        s.commit()

    # Optional: start the continuous labeling research factory.
    if settings.LABELING_AUTO_FETCH_ENABLED:
        try:
            _labeling_pipeline = LabelingPipeline()
            _labeling_pipeline.start()
        except Exception:
            _labeling_pipeline = None

    # Orchestrator is optional; default off for API-only deployments/tests.
    if settings.START_ORCHESTRATOR:
        try:
            _orchestrator = Orchestrator()
            _orchestrator_task = asyncio.create_task(_orchestrator.run())
        except Exception:
            # Keep API up even if orchestrator init fails.
            _orchestrator = None
            _orchestrator_task = None


@app.on_event("shutdown")
async def _shutdown() -> None:
    global _orchestrator, _orchestrator_task, _labeling_pipeline

    if _labeling_pipeline:
        _labeling_pipeline.stop()

    if _orchestrator:
        _orchestrator.stop()

    if _orchestrator_task:
        _orchestrator_task.cancel()
        try:
            await _orchestrator_task
        except Exception:
            # ignore cancel/teardown errors
            pass
