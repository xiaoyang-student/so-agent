"""全局运行配置（Pydantic Settings）。

所有可调参数集中于此，支持环境变量覆盖：
- 环境变量前缀为 ``SO_AGENT_``，例如 ``SO_AGENT_MAX_CONCURRENCY=5``；
- 也可通过项目根目录下的 ``.env`` 文件提供同名变量；
- 未提供的参数使用下方默认值。

治理类参数与架构计划的约束一一对应（动态 Agent 上限、替换次数上限、
重规划次数上限等），编排器运行时应始终通过 ``get_settings()`` 读取，
不得在业务代码中硬编码。
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """so-agent 全局配置。"""

    model_config = SettingsConfigDict(
        env_prefix="SO_AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # 允许 model_timeout 等以 model_ 开头的配置字段，
        # 禁用 pydantic 对 model_ 前缀的保护命名空间警告。
        protected_namespaces=(),
    )

    # ---- 动态 Agent 与替换治理 ----
    max_dynamic_agents: int = 8          # 动态创建的子 Agent 数量上限
    max_agent_attempts: int = 3          # 单个 Agent 的最大尝试次数
    max_replacement_requests: int = 1    # 替换请求次数上限（含动态子 Agent 无替换权约束）

    # ---- 编排与并发 ----
    max_concurrency: int = 3             # 同时在执行的最大子任务数
    max_replan_attempts: int = 3         # 主 Agent 修订任务分解的最大次数
    max_model_turns: int = 10            # 单次 Agent 运行的最大模型回合数

    # ---- 超时与沙箱 ----
    model_timeout: float = 60.0          # 单次模型调用超时（秒）
    sandbox_timeout: float = 30.0        # 沙箱单次命令执行超时（秒）
    sandbox_output_limit: int = 1_048_576  # 沙箱输出上限（字节，默认 1 MiB）


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回进程内唯一的全局配置实例。"""
    return Settings()
