"""工具注册表（ToolRegistry）：MCP 工具注册中心。

职责与行为约定：
- 所有权归主管 Agent：主管 Agent 与控制层可查询工具清单、确认授权候选；
- 所有工具统一按 MCP 规范登记：每个工具附带 MCP 暴露契约
  （``MCPToolSchema``）与对应的 ``MCPServer``——
  ``code_agent`` / ``subagent_creator`` / ``review_agent`` 三个固定入口
  与预置可分配工具（read_file / write_file / run_in_sandbox）在装配阶段
  由 orchestrator 通过 ``attach_mcp`` 挂载真实 MCP Server；动态生成工具
  在注册时自动创建 ``MCPToolServer``（惰性加载源码）；
- ``assignable`` 属性：``code_agent`` 与 ``subagent_creator`` 为主管专属
  工具（assignable=False），严禁分配给任何子 Agent；其余工具可分配；
- 动态生成工具（GeneratedToolManifest）：只有 ``review_status ==
  "approved"`` 才可放行使用；未评审或未通过的工具注册后仅保留元数据，
  ``is_approved`` 返回 False；
- 支持从 ``generated_tools/registry.json`` 批量加载与写回。

全局约束：本表只描述工具元数据与 MCP 暴露契约，不代表调用许可；任何
实际调用必须经 PermissionGateway 的 ToolGrant 校验后才能执行。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from so_agent.mcp.adapter import MCPAdapterError, generated_tool_to_mcp_server
from so_agent.mcp.protocol import MCPToolSchema
from so_agent.mcp.server import MCPServer
from so_agent.models import GeneratedToolManifest

# 三个固定工具入口的默认注册信息：工具名 →（描述, 工具包引用）
DEFAULT_BUILTIN_AGENTS: dict[str, tuple[str, str]] = {
    "code_agent": (
        "代码执行 Agent：在验证型沙箱内编写并运行代码，返回 ExecutionResult",
        "so_agent.tool_packages.code_agent",
    ),
    "subagent_creator": (
        "子 Agent 创建器：按需动态创建专用子 Agent 并登记 AgentRecord",
        "so_agent.tool_packages.subagent_creator",
    ),
    "review_agent": (
        "评审 Agent：执行计划评审与结果评审，返回 ReviewDecision",
        "so_agent.tool_packages.review_agent",
    ),
}

# 主管专属工具：不可分配给任何子 Agent（issue_grant 会直接拒绝）
SUPERVISOR_EXCLUSIVE_TOOLS: frozenset[str] = frozenset(
    {"code_agent", "subagent_creator"}
)

# 预置可分配工具：子 Agent 可经授权获得的基础能力（与 code_agent 内部
# 工具同名同源）；注册为内置工具，使"工具缺口检测"（未注册即缺口）
# 对全部工具名保持统一语义。
DEFAULT_ASSIGNABLE_TOOLS: dict[str, tuple[str, dict[str, Any]]] = {
    "read_file": (
        "读取项目沙箱目录内的文本文件",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对沙箱根目录的文件路径"}
            },
            "required": ["path"],
        },
    ),
    "write_file": (
        "在项目沙箱目录内写入（覆盖）一个文本文件",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对沙箱根目录的文件路径"},
                "content": {"type": "string", "description": "要写入的完整文本内容"},
            },
            "required": ["path", "content"],
        },
    ),
    "run_in_sandbox": (
        "在验证型沙箱中运行一段 Python 代码并返回结构化结果",
        {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "待执行的 Python 源码"},
                "task_id": {"type": "string", "description": "任务编号"},
            },
            "required": ["code", "task_id"],
        },
    ),
}

# 预置工具的工具包引用（与 code_agent 内部工具同源）
_PRESET_AGENT_REF = "so_agent.tool_packages.code_agent"

# generated_tools 目录默认位置（相对本模块定位，避免导入 tool_packages）
DEFAULT_GENERATED_TOOLS_DIR: Path = (
    Path(__file__).resolve().parents[1] / "tool_packages" / "generated_tools"
)


class ToolRegistryError(Exception):
    """工具注册表领域错误：重名冲突、清单非法、文件读写失败等。"""


class ToolRegistry:
    """工具注册表（内存实现 + registry.json 加载/写回 + MCP 契约挂载）。"""

    def __init__(
        self,
        *,
        builtin_agents: Mapping[str, tuple[str, str]] | None = None,
        generated_registry_path: str | Path | None = None,
        generated_tools_dir: str | Path | None = None,
        include_assignable_presets: bool = True,
    ) -> None:
        """初始化注册表。

        Args:
            builtin_agents: 固定工具入口覆盖配置（默认使用
                ``DEFAULT_BUILTIN_AGENTS``）。
            generated_registry_path: 可选；提供时初始化即加载该 registry.json。
            generated_tools_dir: 生成工具源码所在目录（默认
                ``tool_packages/generated_tools``）；注册生成工具时据此
                自动创建 MCPToolServer。
            include_assignable_presets: 是否注册预置可分配工具
                （read_file / write_file / run_in_sandbox），默认 True。
        """
        self._tools: dict[str, dict[str, Any]] = {}
        self._manifests: dict[str, GeneratedToolManifest] = {}
        self._approved: set[str] = set()
        self._mcp_schemas: dict[str, MCPToolSchema] = {}
        self._mcp_servers: dict[str, MCPServer] = {}
        self._generated_tools_dir = (
            Path(generated_tools_dir)
            if generated_tools_dir is not None
            else DEFAULT_GENERATED_TOOLS_DIR
        )

        source = builtin_agents if builtin_agents is not None else DEFAULT_BUILTIN_AGENTS
        for name, (description, agent_ref) in source.items():
            self.register_builtin_tool(name, description, agent_ref)

        if include_assignable_presets:
            for name, (description, input_schema) in DEFAULT_ASSIGNABLE_TOOLS.items():
                self.register_builtin_tool(
                    name,
                    description,
                    _PRESET_AGENT_REF,
                    assignable=True,
                    input_schema=input_schema,
                )

        if generated_registry_path is not None:
            self.load_generated_registry(generated_registry_path)

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------
    def register_builtin_tool(
        self,
        name: str,
        description: str,
        agent_ref: str,
        *,
        assignable: bool | None = None,
        input_schema: dict[str, Any] | None = None,
        mcp_server: MCPServer | None = None,
    ) -> None:
        """注册（或覆盖）一个固定内置工具入口，并登记其 MCP 契约。

        Args:
            name: 工具名称；``code_agent`` / ``subagent_creator`` 会被
                强制标记为不可分配（主管专属工具）。
            description: 工具用途说明（写入 MCPToolSchema）。
            agent_ref: 工具包引用（如 ``so_agent.tool_packages.code_agent``）。
            assignable: 是否可分配给子 Agent；None 时按主管专属工具清单
                自动推导（专属 → False，其余 → True）。
            input_schema: 输入参数 JSON Schema（MCP inputSchema）。
            mcp_server: 可选的 MCP Server（装配阶段由 orchestrator 挂载）。

        Raises:
            ToolRegistryError: 名称 / 引用为空，或 mcp_server 类型非法时。
        """
        if not isinstance(name, str) or not name.strip():
            raise ToolRegistryError("内置工具名称不能为空")
        if not isinstance(agent_ref, str) or not agent_ref.strip():
            raise ToolRegistryError(f"内置工具 {name!r} 的工具包引用不能为空")
        if assignable is None:
            assignable = name not in SUPERVISOR_EXCLUSIVE_TOOLS
        self._tools[name] = {
            "name": name,
            "description": description,
            "agent_ref": agent_ref,
            "source": "builtin",
            "approved": True,  # 固定入口与预置工具视为已批准
            "assignable": bool(assignable),
        }
        self._mcp_schemas[name] = MCPToolSchema(
            name=name,
            description=description,
            inputSchema=(
                dict(input_schema)
                if isinstance(input_schema, dict) and input_schema
                else {"type": "object", "properties": {}}
            ),
        )
        if mcp_server is not None:
            if not isinstance(mcp_server, MCPServer):
                raise ToolRegistryError(
                    f"mcp_server 必须是 MCPServer，实际为 {type(mcp_server).__name__}"
                )
            self._mcp_servers[name] = mcp_server

    def register_generated_tool(self, manifest: GeneratedToolManifest) -> None:
        """注册（或更新）一个动态生成工具，并自动创建对应 MCPToolServer。

        同名生成工具重复注册视为版本更新；与内置固定入口重名则拒绝。

        Raises:
            ToolRegistryError: manifest 类型非法，或与内置工具重名时。
        """
        if not isinstance(manifest, GeneratedToolManifest):
            raise ToolRegistryError(
                f"manifest 必须是 GeneratedToolManifest，实际为 {type(manifest).__name__}"
            )
        name = manifest.tool_name
        existing = self._tools.get(name)
        if existing is not None and existing.get("source") == "builtin":
            raise ToolRegistryError(f"生成工具 {name!r} 与内置固定工具重名，拒绝注册")

        self._manifests[name] = manifest
        entry = manifest.model_dump(mode="json")
        entry["source"] = "generated"
        entry["approved"] = manifest.review_status == "approved"
        entry["assignable"] = True  # 生成工具可分配（以 approved 状态为门槛）
        self._tools[name] = entry
        if entry["approved"]:
            self._approved.add(name)
        else:
            self._approved.discard(name)

        # MCP 契约：优先取 manifest.mcp_schema，回退 input_schema / 空对象
        self._mcp_schemas[name] = self._schema_from_manifest(manifest)
        # 注册即自动创建 MCP Server（handler 惰性加载源码，调用时才读盘）
        try:
            self._mcp_servers[name] = generated_tool_to_mcp_server(
                self._generated_tools_dir / manifest.entry_file, manifest
            )
        except MCPAdapterError:
            self._mcp_servers.pop(name, None)

    @staticmethod
    def _schema_from_manifest(manifest: GeneratedToolManifest) -> MCPToolSchema:
        """从生成工具清单构造 MCP 暴露契约（宽容解析 mcp_schema 字段）。"""
        raw = dict(manifest.mcp_schema or {})
        input_schema = raw.get("inputSchema")
        if not isinstance(input_schema, dict) or not input_schema:
            input_schema = manifest.input_schema
        if not isinstance(input_schema, dict) or not input_schema:
            input_schema = {"type": "object", "properties": {}}
        return MCPToolSchema(
            name=manifest.tool_name,
            description=str(raw.get("description") or manifest.description or ""),
            inputSchema=dict(input_schema),
        )

    # ------------------------------------------------------------------
    # MCP 契约挂载与查询
    # ------------------------------------------------------------------
    def attach_mcp(
        self,
        tool_name: str,
        schema: MCPToolSchema,
        server: MCPServer | None = None,
    ) -> None:
        """为已注册工具挂载（或更新）MCP 暴露契约与 MCP Server。

        装配阶段由 orchestrator 调用：把真实包装好的 MCP Server 挂到
        对应固定工具入口上，供主管 Agent 与 MCPClient 统一发现/调用。

        Raises:
            ToolRegistryError: 工具未注册或参数类型非法时。
        """
        if tool_name not in self._tools:
            raise ToolRegistryError(f"工具未注册，无法挂载 MCP 契约：{tool_name!r}")
        if not isinstance(schema, MCPToolSchema):
            raise ToolRegistryError(
                f"schema 必须是 MCPToolSchema，实际为 {type(schema).__name__}"
            )
        self._mcp_schemas[tool_name] = schema
        if server is not None:
            if not isinstance(server, MCPServer):
                raise ToolRegistryError(
                    f"server 必须是 MCPServer，实际为 {type(server).__name__}"
                )
            self._mcp_servers[tool_name] = server

    def get_mcp_schema(self, tool_name: str) -> MCPToolSchema | None:
        """返回工具的 MCP 暴露契约；未注册或未挂载返回 None。"""
        return self._mcp_schemas.get(tool_name)

    def get_mcp_server(self, tool_name: str) -> MCPServer | None:
        """返回工具对应的 MCP Server；未挂载返回 None。"""
        return self._mcp_servers.get(tool_name)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def get_tool(self, name: str) -> dict[str, Any] | None:
        """按名称返回工具元数据（浅拷贝）；不存在返回 None。"""
        entry = self._tools.get(name)
        return dict(entry) if entry is not None else None

    def list_tool_names(self) -> list[str]:
        """返回全部已注册工具名称（按字典序）。"""
        return sorted(self._tools)

    def list_tools(self) -> list[MCPToolSchema]:
        """返回全部已注册工具的 MCP 暴露契约（按工具名字典序）。"""
        return [self._mcp_schemas[name] for name in sorted(self._tools)]

    def is_approved(self, tool_name: str) -> bool:
        """判断工具是否可放行使用。

        内置固定入口与预置工具恒为 True；生成工具需 ``review_status ==
        "approved"``；未注册的工具返回 False。
        """
        entry = self._tools.get(tool_name)
        if entry is None:
            return False
        return bool(entry.get("approved"))

    def list_approved_tools(self) -> list[str]:
        """返回全部可放行的工具名称（按字典序）。"""
        return [name for name in self.list_tool_names() if self.is_approved(name)]

    def is_assignable(self, tool_name: str) -> bool:
        """判断工具是否可分配给子 Agent（主管专属工具返回 False）。"""
        entry = self._tools.get(tool_name)
        return bool(entry is not None and entry.get("assignable", False))

    def list_assignable_tools(self, *, approved_only: bool = False) -> list[dict[str, Any]]:
        """返回可分配工具清单（按字典序；元素为 JSON 兼容的元数据摘要）。

        Args:
            approved_only: 为 True 时仅返回已批准（可放行）的工具，
                用于构造规划阶段的 ``available_tools`` 与子 Agent 注入候选。
        """
        items: list[dict[str, Any]] = []
        for name in self.list_tool_names():
            entry = self._tools[name]
            if not entry.get("assignable", False):
                continue
            approved = bool(entry.get("approved", False))
            if approved_only and not approved:
                continue
            items.append(
                {
                    "name": name,
                    "description": entry.get("description", ""),
                    "source": entry.get("source"),
                    "approved": approved,
                }
            )
        return items

    def get_manifest(self, tool_name: str) -> GeneratedToolManifest | None:
        """返回生成工具的原始清单对象（供审计/写回）；内置工具或无记录返回 None。"""
        return self._manifests.get(tool_name)

    # ------------------------------------------------------------------
    # registry.json 加载与写回
    # ------------------------------------------------------------------
    def load_generated_registry(self, path: str | Path) -> int:
        """从 registry.json 批量加载生成工具清单，返回成功加载的条数。

        Raises:
            ToolRegistryError: 文件不存在、JSON 非法、顶层不是数组，
                或任一元素无法解析为 GeneratedToolManifest 时。
        """
        registry_path = Path(path)
        if not registry_path.is_file():
            raise ToolRegistryError(f"生成工具清单不存在：{registry_path}")
        try:
            raw = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ToolRegistryError(
                f"生成工具清单读取/解析失败：{registry_path}（{exc}）"
            ) from exc
        if not isinstance(raw, list):
            raise ToolRegistryError(f"生成工具清单顶层必须是数组：{registry_path}")

        loaded = 0
        for index, item in enumerate(raw):
            try:
                manifest = GeneratedToolManifest.model_validate(item)
            except Exception as exc:  # pydantic ValidationError 等
                raise ToolRegistryError(
                    f"生成工具清单第 {index} 条非法：{registry_path}（{exc}）"
                ) from exc
            self.register_generated_tool(manifest)
            loaded += 1
        return loaded

    def save_generated_registry(self, path: str | Path) -> None:
        """将全部生成工具清单写回 registry.json（按 tool_name 排序，UTF-8）。

        Raises:
            ToolRegistryError: 写入失败时。
        """
        registry_path = Path(path)
        try:
            registry_path.parent.mkdir(parents=True, exist_ok=True)
            payload = [
                self._manifests[name].model_dump(mode="json")
                for name in sorted(self._manifests)
            ]
            registry_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise ToolRegistryError(f"生成工具清单写回失败：{registry_path}（{exc}）") from exc
