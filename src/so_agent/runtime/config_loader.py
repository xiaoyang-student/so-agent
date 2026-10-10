"""api_config.yaml 配置加载器（Config Loader）。

职责与行为约定：
- 读取各工具包目录下的 ``api_config.yaml``，返回结构化的 ``AgentAPIConfig``；
- 密钥配置：支持在 ``api_key`` 字段中直接存放密钥值（优先），同时
  向后兼容旧的 ``api_key_env`` 字段（只声明环境变量名，运行时通过
  ``os.environ`` 解析）；密钥值序列化时自动排除，绝不写入日志；
- 校验必填字段（provider / model，且 api_key / api_key_env 至少
  配置一个），缺失或类型非法时抛出明确的 ``ConfigLoadError``，
  错误信息包含文件路径与缺失字段清单。

调用示例::

    config = load_agent_api_config(".../code_agent/api_config.yaml")
    api_key = config.resolve_api_key()          # 优先 api_key 直接值
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError


class ConfigLoadError(Exception):
    """api_config.yaml 缺失、格式非法或必填字段缺失时抛出。"""


class AgentAPIConfig(BaseModel):
    """单个 Agent 的 API 连接配置。

    字段来源：``provider`` / ``model`` / ``base_url`` / ``timeout`` /
    ``retry`` / ``api_key`` / ``api_key_env`` 均来自 api_config.yaml。
    密钥支持两种配置方式（至少其一）：

    - ``api_key``：直接存放密钥值（优先）；
    - ``api_key_env``：密钥所在的环境变量名（向后兼容，运行时由
      ``resolve_api_key()`` 从 ``os.environ`` 解析后写入 ``api_key``）。

    ``api_key`` 在序列化时自动排除，防止密钥值意外落入日志。
    """

    provider: str
    model: str
    base_url: str | None = None                     # None 表示使用 provider 默认地址
    timeout: float = 60.0                           # 单次请求超时（秒）
    retry: int = 2                                  # 失败自动重试次数
    # 模型 HTTP 客户端的分维度超时（任务 45，全部可选、向后兼容）：
    # 任一维度显式配置时，resolve_model_http_timeout 会构造
    # openai.Timeout（总预算仍取 ``timeout``）；全部未配置时保持旧
    # 行为（单一总超时），既有 api_config.yaml 无需任何修改。
    connect_timeout: float | None = None            # TCP 连接建立超时（秒）
    read_timeout: float | None = None               # 读取超时（两次读之间的间隔上限，非总时长）
    write_timeout: float | None = None              # 写入超时（秒）
    pool_timeout: float | None = None               # 连接池获取超时（秒）
    api_key: str | None = Field(default=None, exclude=True)  # 直接密钥值（优先）；序列化时排除
    api_key_env: str | None = None                  # 密钥所在的环境变量名（向后兼容回退）

    def resolve_api_key(self) -> str:
        """解析实际密钥：优先 ``api_key`` 直接值，回退 ``api_key_env`` 环境变量。

        Raises:
            ConfigLoadError: ``api_key`` 为空且 ``api_key_env`` 未配置，
                或 ``api_key_env`` 对应环境变量未设置/为空时。
        """
        direct = (self.api_key or "").strip()
        if direct:
            return direct
        if self.api_key_env:
            value = os.environ.get(self.api_key_env, "")
            if value.strip():
                return value.strip()
            raise ConfigLoadError(
                f"环境变量 {self.api_key_env!r} 未设置或为空，"
                f"无法解析 provider={self.provider!r} 的 API 密钥；"
                "请在 api_config.yaml 中直接配置 api_key，或设置该环境变量。"
            )
        raise ConfigLoadError(
            f"无法解析 provider={self.provider!r} 的 API 密钥："
            "api_config.yaml 中 api_key（直接密钥值）与 api_key_env"
            "（环境变量名）至少需要配置一个。"
        )


# api_config.yaml 中必须存在的字段
REQUIRED_FIELDS: tuple[str, ...] = ("provider", "model")

# 密钥字段：api_key（直接值）与 api_key_env（环境变量名）至少需要一个
SECRET_FIELDS: tuple[str, ...] = ("api_key", "api_key_env")

# 工具包目录下的固定配置文件名
DEFAULT_CONFIG_FILENAME = "api_config.yaml"


def resolve_model_http_timeout(config: "AgentAPIConfig") -> Any:
    """把 AgentAPIConfig 解析为模型 HTTP 客户端的 ``timeout`` 参数。

    任务 45 的可测试 adapter：

    - 四个分维度（connect / read / write / pool）均未配置时，返回
      ``float(config.timeout)``——与旧行为完全一致（完全向后兼容）；
    - 任一维度显式配置时，返回 ``openai.Timeout(timeout, **已配置维度)``
      （未配置的维度回退到总预算值；openai 3.26.0 的 ``Timeout``
      构造器不支持部分维度缺省时仅设置该维度，总预算位置参数必须
      显式给出）。

    边界事实（经 SDK 源码核查，不凭记忆假设）：

    - OpenAI Agents SDK 0.22.3 的 ``OpenAIChatCompletionsModel`` 构造
      ``chat.completions.create`` 调用参数时不传递
      ``ModelSettings.timeout``（仅透传 temperature/top_p 等），因此
      超时只能在 ``AsyncOpenAI`` 客户端级别注入，本函数即该注入点；
    - httpx 的 read timeout 是单次 I/O 操作（流式下为 chunk 间隔）
      的上限而非整次请求总时长；总时长预算由上层兜底
      （run_agent_streamed 的 agent_total 与 pipeline deadline），
      客户端维度超时只负责快速失败，不承担总预算职责。

    Raises:
        ConfigLoadError: 任一分维度配置为非正数时（fail fast，避免
            非法配置静默生效）。
    """
    dimensions: dict[str, float | None] = {
        "connect": config.connect_timeout,
        "read": config.read_timeout,
        "write": config.write_timeout,
        "pool": config.pool_timeout,
    }
    configured = {
        name: float(value)
        for name, value in dimensions.items()
        if value is not None
    }
    invalid = {name: value for name, value in configured.items() if value <= 0.0}
    if invalid:
        raise ConfigLoadError(
            f"模型 HTTP 分维度超时必须为正数：{invalid}"
            f"（provider={config.provider!r}）"
        )
    if not configured:
        return float(config.timeout)
    try:
        from openai import Timeout
    except ImportError as exc:  # pragma: no cover - openai 为硬依赖
        raise ConfigLoadError(
            "openai 包不可用，无法构造分维度超时（Timeout）；"
            "请安装 openai 或移除 connect/read/write/pool_timeout 配置"
        ) from exc
    return Timeout(float(config.timeout), **configured)


def load_agent_api_config(
    config_path: str | Path,
    *,
    resolve_secrets: bool = False,
) -> AgentAPIConfig:
    """读取单个 api_config.yaml 并返回 AgentAPIConfig。

    Args:
        config_path: api_config.yaml 的路径。
        resolve_secrets: 为 True 时立即解析密钥并填入 ``api_key`` 字段
            （优先 api_key 直接值，回退 api_key_env 对应环境变量；均不可
            用时抛错）；默认 False，仅保留原始配置。

    Raises:
        ConfigLoadError: 文件不存在、YAML 解析失败、顶层不是键值映射、
            必填字段缺失/为空（含 api_key 与 api_key_env 均未配置）、
            或字段类型非法时。
    """
    path = Path(config_path)
    if not path.is_file():
        raise ConfigLoadError(f"api_config.yaml 不存在：{path}")

    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"api_config.yaml 解析失败：{path}（{exc}）") from exc
    except OSError as exc:
        raise ConfigLoadError(f"api_config.yaml 读取失败：{path}（{exc}）") from exc

    if not isinstance(raw, dict):
        raise ConfigLoadError(
            f"api_config.yaml 顶层必须是键值映射：{path}"
            f"（实际类型：{type(raw).__name__}）"
        )

    missing = [key for key in REQUIRED_FIELDS if not str(raw.get(key) or "").strip()]
    if missing:
        raise ConfigLoadError(
            f"api_config.yaml 缺少必填字段 {missing}：{path}；"
            f"必填字段完整清单为 {list(REQUIRED_FIELDS)}。"
        )

    if not any(str(raw.get(key) or "").strip() for key in SECRET_FIELDS):
        raise ConfigLoadError(
            f"api_config.yaml 缺少 API 密钥字段 {list(SECRET_FIELDS)}：{path}；"
            "api_key（直接密钥值）与 api_key_env（环境变量名）至少配置一个。"
        )

    try:
        config = AgentAPIConfig(**raw)
    except ValidationError as exc:
        raise ConfigLoadError(f"api_config.yaml 字段类型非法：{path}（{exc}）") from exc

    if resolve_secrets:
        config.api_key = config.resolve_api_key()
    return config


def discover_tool_package_configs(
    tool_packages_dir: str | Path,
    *,
    resolve_secrets: bool = False,
) -> dict[str, AgentAPIConfig]:
    """扫描工具包目录下所有 api_config.yaml，返回 ``{工具包名: 配置}``。

    用于启动阶段一次性加载三个固定工具入口（code_agent / subagent_creator /
    review_agent）的默认配置。不含 api_config.yaml 的子目录（如
    generated_tools）会被自动跳过。

    Raises:
        ConfigLoadError: 工具包根目录不存在，或其中任一配置文件非法时。
    """
    root = Path(tool_packages_dir)
    if not root.is_dir():
        raise ConfigLoadError(f"工具包目录不存在：{root}")

    configs: dict[str, AgentAPIConfig] = {}
    for config_path in sorted(root.glob(f"*/{DEFAULT_CONFIG_FILENAME}")):
        package_name = config_path.parent.name
        configs[package_name] = load_agent_api_config(
            config_path, resolve_secrets=resolve_secrets
        )
    return configs
