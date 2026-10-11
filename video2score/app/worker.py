"""后台排队执行：转谱、补导出，一个接一个跑（都很吃 CPU），界面轮询进度。"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger("video2score")

# run.run 报的阶段（进度文字开头）和它在整条进度里占的位置。逐帧找小节那段按它报的百分比在 5%~80% 之间走。
_STAGES = (
    ("取视频", 0.02),
    ("逐帧找小节", 0.05),
    ("读小节号", 0.80),
    ("清洗", 0.84),
    ("排版", 0.88),
    ("读 TAB 和节奏", 0.90),
    ("导出", 0.96),
)
_FRAMES = re.compile(r"^\s*(\d+)%")
_PERCENT = re.compile(r"(\d+)%")


class Cancelled(Exception):
    pass


@dataclass
class Task:
    kind: str  # convert（转谱）或 export（补导出）
    label: str  # 界面上显示的名字：链接、文件名或作业名
    work: Callable[["Task"], str | None]  # 真正干活的；返回作业 id
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    state: str = "queued"  # queued、running、done、failed、cancelled
    stage: str = "排队中"
    progress: float = 0.0
    job: str | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    cancel: threading.Event = field(default_factory=threading.Event)

    def say(self, text: str) -> None:
        """run.run 的 progress 回调：记下阶段和进度；要取消时在这里打断。"""
        if self.cancel.is_set():
            raise Cancelled("已取消")
        text = text.strip()
        log.info("[%s] %s", self.id, text)
        match = _FRAMES.match(text)
        if match:
            self.progress = 0.05 + 0.75 * int(match.group(1)) / 100
            return
        for index, (name, start) in enumerate(_STAGES):
            if text.startswith(name):
                self.stage = text.replace("……", " ").strip()
                # 阶段里带百分比的（"取视频…… 35%"）：在这一段和下一段的起点之间走。
                percent = _PERCENT.search(text)
                end = _STAGES[index + 1][1] if index + 1 < len(_STAGES) else 1.0
                self.progress = max(self.progress, start + (end - start) * int(percent.group(1)) / 100 if percent else start)
                return

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "state": self.state,
            "stage": self.stage,
            "progress": round(self.progress, 3),
            "job": self.job,
            "error": self.error,
        }


class Worker:
    """一个后台线程，按提交顺序执行任务。只在内存里记任务；作业本身都在作业目录里。"""

    def __init__(self) -> None:
        self._queue: deque[Task] = deque()
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Condition()
        self._thread = threading.Thread(target=self._loop, name="worker", daemon=True)
        self._thread.start()

    def submit(self, task: Task) -> Task:
        with self._lock:
            self._tasks[task.id] = task
            self._queue.append(task)
            self._lock.notify()
        return task

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.state not in ("queued", "running"):
                return False
            task.cancel.set()
            if task.state == "queued":
                self._queue.remove(task)
                task.state, task.stage = "cancelled", "已取消"
            return True

    def tasks(self) -> list[dict]:
        """最近的任务（新的在前）：排队的、在跑的，以及一小时内结束的。"""
        now = time.time()
        with self._lock:
            items = [task for task in self._tasks.values() if task.state in ("queued", "running") or now - task.created < 3600]
        return [task.to_dict() for task in sorted(items, key=lambda task: task.created, reverse=True)]

    def busy(self) -> bool:
        with self._lock:
            return any(task.state in ("queued", "running") for task in self._tasks.values())

    def _loop(self) -> None:
        while True:
            with self._lock:
                while not self._queue:
                    self._lock.wait()
                task = self._queue.popleft()
                task.state, task.stage = "running", "开始"
            try:
                task.job = task.work(task)
                task.state, task.stage, task.progress = "done", "完成", 1.0
            except Cancelled:
                task.state, task.stage = "cancelled", "已取消"
            except Exception as exc:  # 一个任务出错不影响后面的
                log.exception("任务 %s 出错", task.id)
                task.state, task.stage, task.error = "failed", "出错", str(exc)
