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

    # --- iFind / Data Provider ---
    # keep historical naming but actual provider is iFind
    THS_MODE: str = "IFIND_HTTP"  # MOCK / IFIND_HTTP / IFIND_SDK / DATAFEED
    IFIND_HTTP_BASE_URL: str = ""
    IFIND_HTTP_TOKEN: str = ""

    # Dispatcher/provider routing
    DATA_PROVIDER: str = "IFIND_HTTP"  # IFIND_HTTP / MOCK

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

    # ==========================
    # Agent / Swing目标参数（本轮新增）
    # ==========================
    HOLD_DAYS_MIN: int = 1
    HOLD_DAYS_MAX: int = 3

    # 目标 5%~8%
    TARGET_RETURN_MIN: float = 0.05
    TARGET_RETURN_MAX: float = 0.08

    # Agent行为
    AGENT_MAX_REQUESTS_PER_SYMBOL: int = 8
    AGENT_VERIFY_MIN_CONFIDENCE: float = 0.55

    # 结构化证据阈值（偏保守，先防“大亏”）
    AGENT_BREAKOUT_LOOKBACK_DAYS: int = 20  # 20日突破
    AGENT_MIN_DAILY_BARS: int = 25          # 至少要有足够历史
    AGENT_MAX_DAILY_VOL_PROXY: float = 0.08 # 日收益率std（粗略）上限（过大波动先不做）
    AGENT_MIN_MOMENTUM_3D: float = 0.01     # 3日动量最低门槛（先弱约束）
    AGENT_VOLUME_SURGE_MULT: float = 1.5    # 当日量 > 过去均量 * 倍数

    # 盘中确认
    AGENT_INTRADAY_MIN_BARS: int = 20       # 至少20根分钟K
    AGENT_INTRADAY_MAX_DRAWDOWN: float = 0.015  # 盘中最大回撤不超过1.5%
    AGENT_INTRADAY_REQUIRE_ABOVE_VWAP: bool = True


settings = Settings()
