"""
模拟模型生成器（占位）：真实系统应将模型快照写入 ModelSnapshots 并版本冻结。
这里仅提供脚手架，避免项目缺脚本。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from app.utils.crypto import sha256_hex

SH = ZoneInfo("Asia/Shanghai")


def main() -> None:
    now = datetime.now(tz=SH).isoformat()
    weights = {"dummy_weight": 1.0}
    report = {"Net_Return": 0.0, "TailLoss": 0.0, "Trade_Rate_Delta": 0.0}
    blob = json.dumps({"weights": weights, "report": report, "time": now}, sort_keys=True).encode("utf-8")
    model_snapshot_uuid = sha256_hex(blob)[:32]
    print("MODEL_SNAPSHOT_UUID =", model_snapshot_uuid)
    print("weights =", weights)
    print("report =", report)


if __name__ == "__main__":
    main()
