1) 两层过滤规则（建议）
规则 A：市场（Exchange）

allowed_exchanges = ["SZ"]（现在只要深市）

后期要沪市：["SZ","SH"]

规则 B：代码前缀（Prefix）

现在：allowed_prefixes = ["0"]

后期加沪市：allowed_prefixes = ["0","6"]

这样你一句配置就能从“只看 0 开头”扩到“0+6”，而不用动任何业务逻辑。

2) 统一符号规范（避免后面乱）

你接口回的是 code="000593" 这种纯数字串，我们内部统一成：

如果 code 以 0/3 开头 → symbol = code + ".SZ"

如果 code 以 6 开头 → symbol = code + ".SH"

（当然也允许你以后接口直接给 002219.SZ 这种完整 symbol，我们检测到有后缀就不再猜。）

3) 配置放哪（你想“可配置”，我给三种层级，从简到强）
方案 1：环境变量（最简单、上线快）

POOL_ALLOWED_PREFIXES=0

POOL_ALLOWED_EXCHANGES=SZ

后期改成：

POOL_ALLOWED_PREFIXES=0,6

POOL_ALLOWED_EXCHANGES=SZ,SH

方案 2：数据库配置（可在前端动态改）

表：system_settings

key: pool.allowed_prefixes value: ["0"]

key: pool.allowed_exchanges value: ["SZ"]

前端给一个“管理员设置”页面，改完立即生效。

方案 3：策略配置版本化（更严谨）

表：rule_sets（versioned）

每次修改生成新版本，写 effective_ts

pipeline 读取“当天生效版本”

回测/复盘可追溯当时用的是哪套过滤规则

你说“脑要严谨”，方案 3 最符合“可审计”。

4) 对你现有脚本的最小改动思路（概念上）

把这句：

if str(data['code'])[:3] not in r_market_filter: continue


替换成更通用的：

prefix = code[0]

如果 prefix not in allowed_prefixes → skip

根据 prefix 推断 exchange：0/3->SZ, 6->SH

如果 exchange not in allowed_exchanges → skip

并在最终落库时统一写 symbol（带 .SZ/.SH）。

5) “只看 0 开头”对业务意味着什么（小提醒）

0 开头你会错过：

深市 002（中小板历史段）、300（创业板）

沪市 6（你后面要加的）

这没问题，因为你明确说“现在只想看 0 开头”。我会把这当成策略规则的一部分记录进批次元数据里，未来你扩范围时，评估指标（命中率/收益）也能按规则版本对比，不会混账。

下一步我们可以把“候选池过滤规则”直接纳入 batch 元数据（例如 batch 记录里保存当时的 allowed_prefixes/allowed_exchanges），这样你以后复盘某天为什么没出现 6 开头，系统能给出明确解释，而不是靠记忆。

现在可以书写产品文档了吗？把我们讨论的不遗漏的记录下来，然后更具优先级一点点完成
AIStock 决策系统 产品文档 v1.0（讨论纪要 + 可执行路线图）
1. 项目目的与成功标准
1.1 项目目的

在 T+0（当天）收盘后，系统接收一批“待打标候选股票池”（当前来自你提供的接口），并在 T+1 次日 08:30 前完成数据采集与推理，向用户输出：

更适合买哪些（Top N）

推荐理由是什么

推荐依据是什么（可追溯证据）

模型的置信度/风险提示

后续通过 回溯打标 矫正模型与展示真实准确率

用户不关心你抓了多少数据、跑了多少任务；用户只关心：推荐是否更准、为什么准、风险是什么、历史表现如何。

1.2 时间硬约束（统一 Asia/Shanghai）

候选池拉取：每天 16:00

上传/编辑窗口：16:00 之后可编辑，提交后立即进入下一阶段

推理出报告：T+1 08:30 前必须完成

回溯打标：T+1 收盘后（或任意时间补算）

1.3 成功标准（KPI）

系统必须能够持续产出“可量化”的效果指标：

hit_rate@k：TopK 命中率（如次日涨停/强势阈值）

avg_return@k：TopK 平均收益/溢价

drawdown@k：TopK 最大回撤（风险）

coverage：每天可推荐标的数量占比

calibration：概率校准质量（如 Brier score / 分桶命中率）

2. 输入来源与候选池流程
2.1 候选池数据源（由你提供的接口）

系统每天 16:00 准时请求你提供的接口，获取连板/涨停候选池数据。

接口返回样例（来自你给的示例数据）包含：

code（如 000593）

name

high_days（如 3天3板）

limit_up_type（如 换手板）

turnover_rate

order_amount / order_volume

currency_value / sum_market_value

is_again_limit / is_new / change_tag

time_preview[]（日内涨幅轨迹）

hash_key、date 等

2.2 过滤规则（可配置）

当前需求：只看 0 开头；后期可能扩到 6 开头。

设计为配置项，而非写死：

allowed_prefixes：默认 ["0"]，未来可改为 ["0","6"]

allowed_exchanges：默认 ["SZ"]，未来可改为 ["SZ","SH"]

符号规范化规则（canonical）：

0/3 开头 → symbol = code + ".SZ"

6 开头 → symbol = code + ".SH"

若接口未来直接给 002219.SZ，则检测到后缀就直接采用

2.3 预留涨停概率字段（可人工编辑）

候选池条目上必须预留 p_limit_up 字段，供你编辑/粘贴/导入。

p_limit_up 可为空（编辑前）

一旦你编辑完成并 提交（commit），系统立刻进入下一阶段（采集与推理）

3. 总体架构与可插拔设计（微服务化）
3.1 总原则

模型/推理服务只认内部“稳定数据契约（Canonical Schema）”

外部数据源（iFinD、你这个接口、未来任意平台）都通过 Adapter 映射到同一套落库结构

未来若增加自动下单，必须作为独立服务 插拔，不影响推理链路

3.2 推荐服务拆分（最小合理微服务）

pool-fetcher-service

每天 16:00 拉候选池接口

做过滤（0/6 可配置）

写入候选池批次与条目（含 raw）

pool-editor-ui（前端页面）

展示当天批次候选池

支持编辑/导入 p_limit_up

一键 commit 提交进入下一阶段

collector-service（可替换数据源的采集服务）

接到 COMMITTED 批次后：按模块需求拉取数据（iFinD 或其他）

写入“规范化模块表” + 原始归档

feature-service

从模块表抽特征

写 feature 快照（版本化）

reasoning-service（脑）

输出 model_decisions + decision_evidence

生成用户报告（可供前端展示）

labeling-service（回溯与评估）

收盘后计算真实标签与效果指标

写 metrics，供前端展示“准确率/收益/校准”

（未来）execution-service（自动下单）

订阅决策结果

完全不影响前面链路

3.3 明确删除/不做

本轮整改明确：不做自动下单，不保留交易执行链路（可未来以微服务方式加回）。

4. 核心数据契约（Canonical Schema v1）

任何平台数据源只要能把这些字段落库，模型与前端无需变更。

4.1 批次与候选池（输入层）

limitup_pool_batches

batch_id

trading_day（T+0）

source

fetch_ts

status: FETCHED / EDITING / COMMITTED / CANCELLED

filter_rules（保存当时的 allowed_prefixes/exchanges，方便审计与复盘）

raw_hash

limitup_candidates

batch_id

symbol

name

p_limit_up（可空，编辑后填入）

p_source（MANUAL/THS/…）

edited_ts

candidate_status: PENDING_EDIT / READY / DROPPED

raw_json

4.2 数据采集审计层（建议保留）

data_requests / data_responses

可追溯每次采集请求、响应、错误、哈希、时间戳

为“换源/复跑/排障”提供审计

4.3 规范化模块表（推理依赖）

模块1：EOD 行情
equity_eod_snapshot

trading_day, symbol

prev_close/open/high/low/close

volume/amount

turnover_rate

amplitude

float_market_cap

is_limit_up_close

模块2：涨停质量画像
equity_limitup_profile

trading_day, symbol

first_seal_time/last_seal_time

seal_duration_sec

break_count

seal_order_amount

is_one_word_limitup

模块3：资金流
equity_moneyflow_snapshot

trading_day, symbol

net_inflow_total

net_inflow_main

net_inflow_super_large

net_inflow_large

模块4：题材/板块
equity_theme_map

trading_day, symbol, theme_id, theme_name, theme_rank

theme_daily_stats

trading_day, theme_id

theme_strength_score

limitup_count_in_theme

模块5：市场情绪
market_sentiment_daily

trading_day

limitup_count

broken_limitup_count

broken_rate

advancers/decliners

模块6：风险旗标（硬 veto）
equity_risk_flags

trading_day, symbol

is_st

has_major_unlock

has_regulatory_notice

has_suspend_risk

risk_level（LOW/MID/HIGH）

risk_reasons（枚举列表）

模块7：隔夜事件/公告
equity_events

event_id, symbol

event_time

event_type

title

severity

source

raw_ref

4.4 特征层（版本化）

feature_snapshots

trading_day, asof_ts, symbol

schema_version（CANON v1）

feature_set_version

features_json

lineage（关联 request_ids/response_hashes）

4.5 决策输出（用户答案）

model_decisions

trading_day（T+0）

decision_day（T+1）

symbol

action: BUY/WATCH/AVOID

score（0–100）

confidence（0–1）

created_ts（必须 ≤ T+1 08:30）

decision_evidence

decision_id

reason_code

reason_text（短句）

evidence_fields（关键字段值）

evidence_refs（事件/快照引用）

4.6 回溯打标与评估

decision_labels

decision_id

label_day（T+1）

hit_limitup（bool）

close_return/max_return/drawdown

error_tags[]（错因标签）

model_metrics_daily

trading_day

hit_rate@k

avg_return@k

drawdown@k

coverage

brier_score/校准相关统计

5. 核心业务流程（状态机）
5.1 每日主流程

16:00 pool-fetcher 拉候选池 → 写批次（FETCHED）+ 条目（PENDING_EDIT）

前端展示候选池 → 你编辑 p_limit_up

你点击 commit：

校验概率范围、至少 N 条已填

批次状态 → COMMITTED

collector 立刻开始采集模块数据（可换源）

feature-service 抽特征快照

reasoning-service 输出决策 + 证据（必须在次日 08:30 前完成）

T+1 收盘后 labeling-service 回填真实结果，更新 metrics

5.2 降级策略（必须有）

若部分模块采集失败：仍需输出决策，但降低 confidence 并在 evidence 中标注“缺失项”

若核心必需字段缺失严重：该票降级为 WATCH/AVOID，并说明原因

6. 推理系统（“脑”）v1 设计原则
6.1 输入信号

你编辑的 p_limit_up（外部概率/主观概率/第三方模型概率），作为强特征

行情、封板质量、资金、题材、情绪、风险、公告等结构化证据

6.2 推理输出必须满足

可解释：每条推荐至少 3 条理由（reason_code + reason_text）

可追溯：每条理由都能映射到字段或事件 ID

可量化：输出 score、confidence、风险提示

6.3 v1 算法建议（先严谨后复杂）

先采用：规则 veto + 可解释加权评分 + 概率校准（后续用回溯数据做）

后续可升级：学习排序（LTR）、更强的概率校准、相似样本检索

7. 前端展示（用户只看价值，不看任务）
7.1 盘前主页（T+1）

推荐列表 TopN：action + score + confidence

每只票的详情：

推荐理由（3–5 条）

推荐依据（字段卡片/事件卡片）

风险提示（明确列出）

7.2 可信度页面（核心卖点）

最近 5/20/60 日 hit_rate@k / avg_return@k / drawdown@k

覆盖率（每天推荐几只）

概率校准表现（分桶命中）

7.3 回溯复盘页（T+1 收盘后）

昨日推荐结果：命中/收益/回撤

错因标签统计：帮助你和系统一起迭代

8. 与数据源（iFinD 等）的关系

iFinD HTTP API 能覆盖主干数据能力：历史行情、高频/快照、基础数据、公告、交易日历、代码映射等，可作为 collector-service 的首选数据源。

未来更换数据源：仅替换 Adapter，不改 canonical 表、不改推理服务、不改前端。

9. 交付路线图（按优先级逐步完成）

目标是“每一步可上线、可产生价值、可测量”，避免一次性大爆炸。

P0（必须先做，跑通闭环）

候选池批次入库（16:00 拉取 + 过滤可配置）

前端候选池编辑 p_limit_up + commit

commit 触发下一阶段（事件/状态驱动）

reasoning 输出最简决策（先用 p_limit_up + 现有候选池字段做解释）

前端展示推荐清单 + 理由 + 置信度

回溯打标（至少能写入“次日是否涨停/收益”）

展示最基础的 hit_rate@k

P0 的关键：先让系统“能每天产出一份可读的盘前建议”，哪怕采集模块还没全上。

P1（把“依据”做扎实：采集与规范化模块）

collector-service 接入 iFinD（或你指定的第一数据源）

落库模块1/2/3/5（行情/封板质量/资金/情绪）

decision_evidence 里展示这些字段证据

metrics 增加 avg_return@k / drawdown@k / coverage

P2（解释力与稳定性增强）

题材/板块模块（模块4）

风险 veto（模块6）

公告事件模块（模块7）

更严格的降级策略与可观测性（SLA：08:30 前 READY/DEGRADED）

P3（数据发现与模型进化）

概率校准（用回溯标签校准 p_limit_up）

相似样本检索（为理由提供统计依据）

学习排序/更高级模型（可选）

P4（未来可插拔：自动下单）

单独的 execution-service 订阅决策；可随时插拔

10. 本轮整改边界（明确承诺）

本轮以“盘前决策与准确率闭环”为核心

不做自动下单，交易执行相关功能可删除或冻结（后续以微服务插拔方式加回）