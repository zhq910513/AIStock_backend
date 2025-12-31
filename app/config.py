from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ENV: str = "prod"
    TZ: str = "Asia/Shanghai"

    DATABASE_URL: str

    # --- Frozen daily versions ---
    CANONICALIZATION_VERSION: str = "canon_v1"
    RULESET_VERSION_HASH: str = "ruleset_dev_hash_001"
    STRATEGY_CONTRACT_HASH: str = "contract_dev_hash_001"
    MODEL_SNAPSHOT_UUID: str = "stable_model_001"
    COST_MODEL_VERSION: str = "cost_v1"
    FEATURE_EXTRACTOR_VERSION: str = "fx_v1"

    # --- Anti-leakage ---
    EPSILON_MIN_MS: int = 200

    # --- Orchestrator ---
    ORCH_SYMBOLS: str = "000001.SZ,600000.SH"
    ORCH_LOOP_INTERVAL_MS: int = 500
    SCHED_DRIFT_THRESHOLD_MS: int = 250  # drift event threshold

    # --- THS ---
    THS_MODE: str = "MOCK"  # MOCK / IFIND_HTTP / IFIND_SDK / DATAFEED
    IFIND_HTTP_BASE_URL: str = ""
    IFIND_HTTP_TOKEN: str = ""

    # --- S³.1 Differential Audit ---
    SHADOW_LIVE_ENABLED: bool = False
    OLD_STRATEGY_CONTRACT_HASH: str = "contract_old_hash"
    NEW_STRATEGY_CONTRACT_HASH: str = "contract_new_hash"
    CONTRA_POSITION_DELTA_THRESHOLD: float = 0.30
    CONTRA_GUARD_LEVEL_DELTA_THRESHOLD: int = 1

    # --- S³.1 Cross-Source Reconciliation ---
    POST_MARKET_RECONCILE_ENABLED: bool = True
    SOURCE_FIDELITY_K: float = 2.0
    SOURCE_FIDELITY_DEGRADE_DELTA: float = 0.10
    SOURCE_FIDELITY_LOW_SCORE: float = 0.80
    SOURCE_FIDELITY_LOW_STREAK_N: int = 3

    # --- Governance: trade gate ---
    REQUIRE_SELF_CHECK_FOR_TRADING: bool = True
    SELF_CHECK_MAX_AGE_SEC: int = 3600  # 1h

    # --- Governance: reset flow ---
    RESET_REQUIRE_DEVOPS_SIGNATURE: bool = True

    # --- Outbox retry policy (deterministic) ---
    OUTBOX_MAX_ATTEMPTS: int = 12
    OUTBOX_BACKOFF_BASE_MS: int = 250
    OUTBOX_BACKOFF_MAX_MS: int = 30_000


settings = Settings()
