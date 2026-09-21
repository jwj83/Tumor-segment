# 系统配置模块：从环境变量构建评测路径、回调和并发设置。
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    workspace: Path
    answer_root: Path
    log_root: Path
    callback_url: str | None
    pipeline_factory: str | None = None
    callback_timeout_seconds: float = 10.0
    callback_attempts: int = 3
    max_workers: int = 1

    @classmethod
    def from_env(cls) -> "Settings":
        workspace = Path(
            os.environ.get(
                "COMPETITION_WORKSPACE",
                "/2026aicompetition/workspace",
            )
        )
        return cls(
            workspace=workspace,
            answer_root=Path(
                os.environ.get("COMPETITION_ANSWER_ROOT", workspace / "answer")
            ),
            log_root=Path(
                os.environ.get("COMPETITION_LOG_ROOT", workspace / "logs")
            ),
            callback_url=os.environ.get("COMPETITION_CALLBACK_URL") or None,
            pipeline_factory=os.environ.get("COMPETITION_PIPELINE_FACTORY") or None,
            callback_timeout_seconds=float(
                os.environ.get("COMPETITION_CALLBACK_TIMEOUT", "10")
            ),
            callback_attempts=int(
                os.environ.get("COMPETITION_CALLBACK_ATTEMPTS", "3")
            ),
            max_workers=int(os.environ.get("COMPETITION_MAX_WORKERS", "1")),
        )
