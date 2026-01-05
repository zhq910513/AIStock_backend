from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ENV: str = "prod"
    TZ: str = "Asia/Shanghai"

    DATABASE_URL: str

    # --- Profit objective / horizon (1-3 days, 5-8%) ---
    HOLD_DAYS_MIN: int = 1
    HOLD_DAYS_MAX: int = 3
    TARGET_RETURN_MIN: float = 0.05
    TARGET_RETURN_MAX: float = 0.08

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
    ORCH_LOOP_INTERVAL_MS: int = 750
    SCHED_DRIFT_THRESHOLD_MS: int = 250  # drift event threshold

    # --- Data providers ---
    DATA_PROVIDER: str = "IFIND_HTTP"  # MOCK / IFIND_HTTP
    IFIND_HTTP_BASE_URL: str = "https://quantapi.51ifind.com"
    IFIND_HTTP_TOKEN: str = ""  # access_token

    # --- Multi-account (execution plane) ---
    DEFAULT_ACCOUNT_ID: str = "ACC_PRIMARY"
    ACCOUNT_IDS: str = "ACC_PRIMARY"  # comma-separated known accounts
    MAX_CONCURRENT_POSITIONS_PER_ACCOUNT: int = 10

    # --- Governance: trade gate ---
    REQUIRE_SELF_CHECK_FOR_TRADING: bool = True
    SELF_CHECK_MAX_AGE_SEC: int = 3600  # 1h

    # --- Outbox retry policy (deterministic) ---
    OUTBOX_MAX_ATTEMPTS: int = 12
    OUTBOX_BACKOFF_BASE_MS: int = 250
    OUTBOX_BACKOFF_MAX_MS: int = 30_000

    # --- Agentic loop ---
    AGENT_MAX_REQUESTS_PER_SYMBOL: int = 6
    AGENT_VERIFY_MIN_CONFIDENCE: float = 0.60

    # --- API schema versioning ---
    API_SCHEMA_VERSION: str = "1"


settings = Settings()
