"""插件契约与注册表（控制面只依赖本模块，不反向依赖具体能力实现）。

插件按 ``kind`` 分为五类：sandbox / llm / mcp / rules / observability。
清单（manifest）声明身份、版本、入口、能力与权限；注册表按
``(kind, name)`` 保存工厂，消费方通过 :meth:`PluginRegistry.create`
取得 provider。外部插件的加载、启用与摘要校验见 ``plugins.loader`` /
``plugins.manager``。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# 插件 API 主版本：主版本不一致直接拒绝加载（破坏性变更）
PLUGIN_API_VERSION: str = "1"


class PluginKind(str, Enum):
    """插件类别（与目录名、清单 id 前缀一致）。"""

    SANDBOX = "sandbox"
    LLM = "llm"
    MCP = "mcp"
    RULES = "rules"
    OBSERVABILITY = "observability"


PLUGIN_KINDS: tuple[str, ...] = tuple(kind.value for kind in PluginKind)

# 插件可申请的权限（未列出的权限一律拒绝，防止清单越权声明）
KNOWN_PERMISSIONS: frozenset[str] = frozenset(
    {
        "network_egress",
        "filesystem_read",
        "filesystem_write",
        "process_exec",
        "secrets_read",
    }
)

_ID_RE = re.compile(
    r"^(?P<kind>sandbox|llm|mcp|rules|observability)\."
    r"(?P<name>[a-z][a-z0-9_-]{0,63})$"
)
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
_ENTRYPOINT_RE = re.compile(
    r"^[A-Za-z0-9_][A-Za-z0-9_./-]*\.py:[A-Za-z_][A-Za-z0-9_]*$"
)
_SECRET_ENV_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_CAPABILITY_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PluginError(Exception):
    """插件领域错误基类。"""


class PluginManifestError(PluginError):
    """清单非法（字段缺失、格式错误、权限越界等）。"""


class PluginLoadError(PluginError):
    """插件加载失败（路径越界、摘要不匹配、导入异常等）。"""


class PluginNotFoundError(PluginError):
    """请求的插件未注册。"""


def validate_plugin_id(value: str) -> str:
    """校验插件 id（``<kind>.<name>``）并返回规范化结果。"""
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise PluginManifestError(
            f"插件 id 非法：{value!r}；要求 <kind>.<name>，name 为小写字母、"
            "数字、下划线或连字符"
        )
    return value


class PluginManifest(BaseModel):
    """插件清单（``plugins/<kind>/<name>/plugin.yaml``）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    kind: PluginKind
    version: str
    entrypoint: str
    api_version: str = PLUGIN_API_VERSION
    description: str = ""
    capabilities: list[str] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)
    # 仅允许环境变量名，禁止在清单中写明文密钥
    secrets: list[str] = Field(default_factory=list)
    # 入口文件 SHA-256（小写 hex）；启用摘要钉住时必需
    sha256: str = ""
    compatibility: str = ""

    @property
    def name(self) -> str:
        """插件短名（id 中 kind 前缀之后的部分）。"""
        return self.id.split(".", 1)[1]

    @model_validator(mode="after")
    def _validate_manifest(self) -> "PluginManifest":
        validate_plugin_id(self.id)
        if not self.id.startswith(f"{self.kind.value}."):
            raise ValueError(
                f"插件 id {self.id!r} 与 kind {self.kind.value!r} 前缀不一致"
            )
        if not _VERSION_RE.match(self.version):
            raise ValueError(f"版本号非法：{self.version!r}（要求 x.y.z）")
        if self.api_version != PLUGIN_API_VERSION:
            raise ValueError(
                f"插件 API 版本不兼容：{self.api_version!r}，"
                f"内核要求 {PLUGIN_API_VERSION}"
            )
        if (
            not _ENTRYPOINT_RE.match(self.entrypoint)
            or ".." in self.entrypoint
            or "\\" in self.entrypoint
            or self.entrypoint.startswith("/")
        ):
            raise ValueError(
                f"entrypoint 非法：{self.entrypoint!r}；要求 <相对路径>.py:<属性>"
            )
        for capability in self.capabilities:
            if not _CAPABILITY_RE.match(capability):
                raise ValueError(f"能力标识非法：{capability!r}")
        unknown = sorted(set(self.permissions) - KNOWN_PERMISSIONS)
        if unknown:
            raise ValueError(f"未授权的权限声明：{unknown}")
        for env_name in self.secrets:
            if not _SECRET_ENV_RE.match(env_name):
                raise ValueError(
                    f"secrets 只能声明环境变量名：{env_name!r}（禁止明文值）"
                )
        if self.sha256 and not _SHA256_RE.match(self.sha256):
            raise ValueError("sha256 必须是 64 位小写十六进制")
        return self

    @classmethod
    def from_mapping(cls, data: Any) -> "PluginManifest":
        """从字典构造清单；校验失败统一转换为 PluginManifestError。"""
        if not isinstance(data, dict):
            raise PluginManifestError("插件清单必须是映射（YAML/JSON 对象）")
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise PluginManifestError(f"插件清单非法：{exc}") from exc


# 插件工厂：由具体 kind 的消费方约定调用参数（如沙箱的 context /
# sandbox_dir / settings / python_executable），注册表只负责保存与调用。
PluginFactory = Callable[..., Any]


@dataclass(frozen=True)
class PluginEntry:
    """注册表条目：清单 + 工厂。"""

    manifest: PluginManifest
    factory: PluginFactory
    builtin: bool = False


def _coerce_kind(kind: PluginKind | str) -> PluginKind:
    if isinstance(kind, PluginKind):
        return kind
    try:
        return PluginKind(str(kind))
    except ValueError as exc:
        raise PluginError(f"未知插件类别：{kind!r}") from exc


class PluginRegistry:
    """按 ``(kind, name)`` 注册与创建 provider 的进程内注册表。"""

    def __init__(self) -> None:
        self._entries: dict[PluginKind, dict[str, PluginEntry]] = {
            kind: {} for kind in PluginKind
        }

    def register(
        self,
        manifest: PluginManifest,
        factory: PluginFactory,
        *,
        builtin: bool = False,
        replace: bool = False,
    ) -> PluginEntry:
        """注册插件；同类别同名默认拒绝覆盖（显式 replace 才允许）。"""
        if not callable(factory):
            raise PluginError(f"插件 {manifest.id} 的工厂不可调用")
        bucket = self._entries[manifest.kind]
        if manifest.name in bucket and not replace:
            raise PluginError(f"插件重复注册：{manifest.id}")
        entry = PluginEntry(manifest=manifest, factory=factory, builtin=builtin)
        bucket[manifest.name] = entry
        return entry

    def entry(self, kind: PluginKind | str, name: str) -> PluginEntry:
        """返回注册条目；未注册时抛 PluginNotFoundError。"""
        bucket = self._entries[_coerce_kind(kind)]
        entry = bucket.get(name)
        if entry is None:
            raise PluginNotFoundError(
                f"插件未注册：{_coerce_kind(kind).value}.{name}"
            )
        return entry

    def create(self, kind: PluginKind | str, name: str, **kwargs: Any) -> Any:
        """调用插件工厂创建 provider（参数由具体 kind 约定）。"""
        return self.entry(kind, name).factory(**kwargs)

    def names(self, kind: PluginKind | str) -> tuple[str, ...]:
        """返回某类别下已注册的插件短名（排序）。"""
        return tuple(sorted(self._entries[_coerce_kind(kind)]))

    def entries(
        self, kind: PluginKind | str | None = None
    ) -> tuple[PluginEntry, ...]:
        """返回注册条目（可按类别过滤，顺序稳定）。"""
        if kind is None:
            result: list[PluginEntry] = []
            for current in PluginKind:
                result.extend(self._entries[current][name] for name in sorted(self._entries[current]))
            return tuple(result)
        bucket = self._entries[_coerce_kind(kind)]
        return tuple(bucket[name] for name in sorted(bucket))

    def manifests(
        self, kind: PluginKind | str | None = None
    ) -> tuple[PluginManifest, ...]:
        """返回清单列表。"""
        return tuple(entry.manifest for entry in self.entries(kind))

    def builtin_names(self, kind: PluginKind | str) -> Iterable[str]:
        """返回内置插件短名。"""
        return tuple(
            entry.manifest.name for entry in self.entries(kind) if entry.builtin
        )
