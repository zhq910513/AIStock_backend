from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Awaitable, Any


@dataclass
class EventBus:
    """
    异步事件总线：非核心任务（通知/异步日志等）不阻塞交易链路。
    """
    queue: asyncio.Queue

    def __init__(self) -> None:
        self.queue = asyncio.Queue(maxsize=10_000)

    async def publish(self, name: str, payload: dict) -> None:
        await self.queue.put((name, payload))

    async def run(self, handler: Callable[[str, dict], Awaitable[None]]) -> None:
        while True:
            name, payload = await self.queue.get()
            try:
                await handler(name, payload)
            finally:
                self.queue.task_done()
