# app/core 模块说明（模型 & 核心业务）

> 结论先讲清楚：`/app/core` **不全是“模型文件”**。
>
> - 这里的 **core** 指“核心业务能力”：数据采集/标签工厂、交易编排、风控/对账、以及模型训练/推理/推荐。
> - 真正的 **机器学习模型相关** 文件主要在：`model_training.py`、`label_tp3d.py`、`recommender_v2.py`（外加少量特征/证据辅助）。

文档分两部分：
1) 先把“模型从数据到预测到快照”的主链路讲透（你最关心）。
2) 再把 `/core` 下每个文件逐一说明：它是谁、干啥、和哪些表/模块交互。

---

## 1) 模型主链路：从口径 → 标签 → 训练 → 线上快照

### 1.1 关键口径（策略物理定律）
- 决策日：`T`（可以有多个 `cutoff_ts`，比如 15:00/15:30/次日开盘后等）
- 入场价：`entry_px = Open(T+1)`（你能买到的最早参考）
- 最晚持有：`T+3`（最多 3 个交易日可卖）
- 目标：在 `T+1..T+3` 任意一天能否实现 `>= +5%` 或 `>= +8%`（用 High 不是 Close）

### 1.2 标签（Label）
对应实现文件：`app/core/label_tp3d.py`
- 输入：`instrument_id`、`signal_day_T`、`cutoff_ts`
- 依赖数据：`fact_daily_ohlcv`（至少要有 `Open(T+1)` 和 `High(T+1..T+3)`）
- 输出：
  - `label_tp5_3d` / `label_tp8_3d`
  - 可交易性辅助标签（用于过滤/惩罚，不需要前台显示止损）：
    - `label_liquidity_ok`
    - `label_gap_risk`
    - `label_limitup_lock`

### 1.3 训练（LightGBM 双头）
对应实现文件：`app/core/model_training.py`
- 目标（两个二分类头）：
  - `TP5_3D` → 输出 `p_tp5_3d`
  - `TP8_3D` → 输出 `p_tp8_3d`
- 训练数据来源：
  - 候选池（`LabelingCandidate`）提供特征的“原始快照字段”
  - 标签（`ModelTrainingLabel3D`）提供 `tp5/tp8` 真值
- 模型产物：存到 `model_artifact`（`artifact_text` 存 LightGBM Booster string，`feature_list` 存特征顺序）
- 关键能力：
  - `build_features_from_snapshot()`：把候选池字段变成稳定特征向量（带缺失标记）
  - `train_objective_lightgbm()`：训练某个 objective 并写入/激活
  - `predict_proba()` / `predict_contrib()`：线上预测概率与因子贡献（用于 evidence/delta）

### 1.4 线上推荐 & 快照轨迹
对应实现文件：`app/core/recommender_v2.py`
- 先加载 DB 中 active 的 TP5/TP8 两个 Booster（若缺失会回退到启发式 scorer）
- 对每个候选票：
  - 生成特征 → 预测 `p_tp5_3d/p_tp8_3d`
  - 计算 `score = 100*(0.6*p_tp8_3d + 0.4*p_tp5_3d) - penalty(可交易性proxy)`
  - 映射 action：买入/观察/忽略
  - 记录 evidence（含 top contrib）
- 同时为了兼容老 UI：仍会写入 legacy 的 `ModelDecision/DecisionEvidence`

> 你要的“每天概率变化 + 三天轨迹可追溯”，核心就是：每次跑出的结果必须写入 `model_prediction_snapshot`（带 `cutoff_ts`）并且写 `model_snapshot_delta + snapshot_data_dependency`。
>
> 目前 v2 推荐里已经能产出 **evidence（含 contrib）**；如果你后续要求 delta/dependency 更精细（比如对 1mK/五档/逐笔做结构化依赖清单），会在 `snapshot_data_dependency` 这一层继续加。

---

## 2) /app/core 目录文件一览（逐文件说明）

> 说明格式：**它是什么 / 干什么 / 主要输入输出 & 关联模块**。

### __init__.py
- 空初始化文件（标记该目录为 Python package）。

### agent_loop.py
- **它是什么**：一个“智能体循环（PLAN→ACT→OBSERVE→VERIFY→DECIDE）”的骨架。
- **干什么**：对单个标的发起数据请求（行情/历史），通过 `DataRequestDispatcher` 落库，再用 `FeatureExtractor` + `ReasoningEngine` 得出决策。
- **关联**：`data_dispatcher.py`、`feature_extractor.py`、`reasoning.py`、`database.repo`。

### broker_adapter.py
- **它是什么**：券商接口抽象（Protocol）。
- **干什么**：定义 `send_order/query_orders/query_fills`；提供 `MockBrokerAdapter` 用于本地无真实券商的闭环测试。
- **关联**：`execution_router.py`、`outbox.py`、`reconciler.py`。

### canonicalization.py
- **它是什么**：下单“规范化/标准化（承重墙之一）”。
- **干什么**：把价格/数量按 tick/lot 规则离散化，生成 `metadata_hash`，确保幂等/可审计。
- **关联**：`order_manager.py`、`outbox.py`、数据库 order/anchor 相关表。

### collector_pipeline.py
- **它是什么**：数据采集流水线（离线/可重复跑）。
- **干什么**：根据 batch 和 symbols，调用 `data_provider` 拉历史 EOD/行情等，写入快照表（例如 `equity_eod_snapshot` 等）。
- **关联**：`adapters.data_provider`、`database.models`、`orchestrator.py`。

### cutoff_tasks.py
- **它是什么**：15:30 的“cutoff 快照固化任务”。
- **干什么**：把当天候选池字段 materialize 成 `FeatIntradayCutoff`（或类似快照表），用于“防未来函数”的特征截面固化。
- **关联**：`database.models.FeatIntradayCutoff`、调度接口 `/tasks/eod_1530`。

### data_dispatcher.py
- **它是什么**：数据请求执行器。
- **干什么**：执行 DB 中待发的 `DataRequest`，调用 provider，写 `DataResponse`，并把响应解析成特征快照。
- **关联**：`FeatureExtractor`（解析/提取）、`Repo`（写 DB）、`labeling_pipeline.py`（后台循环驱动）。

### differential_audit.py
- **它是什么**：S³.1 Differential Audit 的骨架。
- **干什么**：当开启 shadow 模式时写入对比审计事件（当前是占位实现）。
- **关联**：`Repo.system_events`。

### execution_router.py
- **它是什么**：多账户路由（最小实现）。
- **干什么**：对 symbol 选择 account，并返回对应 broker adapter（目前默认 mock broker）。
- **关联**：`broker_adapter.py`、`Repo.accounts`。

### exit_monitor.py
- **它是什么**：退出/卖出监控（可审计）。
- **干什么**：基于持仓、最新价格与策略参数（contract 或 settings），输出 SELL/HOLD 及理由。
- **关联**：`Repo`、`TradeFill/PortfolioPosition` 等表。

### feature_extractor.py
- **它是什么**：原始数据 → 特征 的解析器。
- **干什么**：从 provider 回包（例如 iFind 结构）提取数值、计算简单统计，并生成 `feature_hash`。
- **关联**：`data_dispatcher.py`、`agent_loop.py`。

### guard.py
- **它是什么**：风控“宪法”（一票否决）。
- **干什么**：根据系统状态（panic_halt、symbol_lock 等）决定是否 veto。
- **关联**：`Repo.system_status`、`Repo.symbol_lock`，以及 `reconciler.py`（孤儿成交触发 orange）。

### label_tp3d.py
- **它是什么**：3D 目标收益标签器（你这套策略的核心口径）。
- **干什么**：按 `entry_px=Open(T+1)`、未来三天 `High` 生成 `tp5/tp8` 标签，并可生成可交易性辅助标签。
- **关联**：`fact_daily_ohlcv`、`model_training_label_3d`、`trading_calendar`。

### labeling_pipeline.py
- **它是什么**：标签工厂后台循环。
- **干什么**：周期性把 watchlist 扩展成 `DataRequest`（planner），并驱动 `DataRequestDispatcher` 处理请求，形成特征与数据积累。
- **关联**：`labeling_planner.py`、`data_dispatcher.py`、`Repo.watchlist`。

### labeling_planner.py
- **它是什么**：标签工厂的“计划生成器”。
- **干什么**：决定“对哪些 symbol、拉哪些指标、用什么 correlation_id”来生成 DataRequests。
- **关联**：`labeling_pipeline.py`。

### limitup_labeler.py
- **它是什么**：涨停池相关的解析/标签辅助。
- **干什么**：把供应商涨停池/高标等字段整理成内部可用的结构（给候选池与推荐用）。
- **关联**：`pool_fetcher`/候选池落库流程、`recommender_v1/v2.py`。

### model_training.py
- **它是什么**：LightGBM 训练 + 推理工具箱（核心模型文件）。
- **干什么**：
  - 生成稳定特征（`build_features_from_snapshot`）
  - 训练 `TP5_3D/TP8_3D` 两个 Booster
  - 存 `model_artifact` 并标记 active
  - 提供 `predict_proba/predict_contrib` 给线上使用
- **关联**：`label_tp3d.py`（生成标签）、`recommender_v2.py`（线上调用）。

### orchestrator.py
- **它是什么**：系统编排器（抓池、落库、跑模型、跑采集、写决策）。
- **干什么**：
  - 拉取涨停池（`fetch_limitup_pool`）
  - 过滤/规范化 symbol
  - 生成 batch
  - 调用 collectors 补数据
  - 调用 `recommender_v2.generate_for_batch_v2` → `persist_decisions_v2`
- **关联**：采集、推荐、数据库 batch/decision 表。

### order_manager.py
- **它是什么**：订单状态机 + 幂等约束入口。
- **干什么**：维护 order state transitions、version_id 乐观锁、写 transitions 记录。
- **关联**：`outbox.py`、`canonicalization.py`、`reconciler.py`。

### outbox.py
- **它是什么**：Outbox 派发器（承重墙之一）。
- **干什么**：把待发送订单从 outbox 发给 broker，写 anchors（RequestUUID/AckHash/RawRequestHash 等），保证可追溯/幂等。
- **关联**：`execution_router.py`、`broker_adapter.py`、`order_manager.py`。

### reasoning.py
- **它是什么**：推理打分引擎（非 ML 版本的“脑子”）。
- **干什么**：对特征做一个确定性 score/confidence 计算，并提供版本绑定 hash。
- **关联**：`agent_loop.py`、系统版本冻结/治理相关 settings。

### recommender_v1.py
- **它是什么**：旧推荐逻辑（基于涨停概率等简化规则）。
- **干什么**：把 `p_limit_up` 映射成 BUY/WATCH/AVOID，并写 legacy 决策表。
- **备注**：当前项目保留它主要用于历史兼容/对照；你的新口径主推 v2。

### recommender_v2.py
- **它是什么**：新推荐逻辑（以 TP5/TP8 目标收益口径为核心）。
- **干什么**：
  - 读取 active LightGBM 双头模型
  - 输出 `p_tp5_3d/p_tp8_3d/score/action`
  - 写 evidence（含贡献度）
  - 兼容写 legacy 决策表（便于 UI 先不改也能用）

### reconciler.py
- **它是什么**：对账引擎（fills-first + 冲突挂起）。
- **干什么**：先写 TradeFill，再反推订单/持仓；发现 ORPHAN/AMBIGUOUS 挂起并告警。
- **关联**：`Guard`、`OrderManager`、订单/成交/持仓表。

### source_fidelity.py
- **它是什么**：跨源对账/源可靠度（S³.1）骨架。
- **干什么**：盘后对账占位逻辑：开启时记录事件，但不改变交易行为。
- **关联**：`Repo.system_events`。

---

## 3) 你问的“/core 下是不是都是模型文件？”——一句话答案
不是。

- `/core` 里 **模型相关** 是一部分（训练、标签、推荐、证据）。
- 另一大部分是 **交易系统骨架**（编排、风控、下单、outbox、对账、审计），以及 **数据工厂**（请求、采集、特征提取）。

如果你后续希望目录更“纯净”，可以把它拆成：
- `core/ml/*`（label/training/recommender/features）
- `core/trading/*`（order/outbox/reconcile/guard/exit）
- `core/pipeline/*`（collector/dispatcher/labeling_pipeline）

目前保持在一个 core 目录里，是为了 P0 快速跑通闭环。
