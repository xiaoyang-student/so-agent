# 外部 MCP Server 目录

把外部 MCP Server 的声明放进本目录，任务开始时自动扫描并注册；无需改代码。

## 目录格式

```
tools/mcp_servers/
  <server-name>/
    server.yaml
```

目录名必须与 `server.yaml` 的 `name` 一致。

## server.yaml

```yaml
name: market-search
enabled: true
transport: https_jsonrpc
url: https://mcp.example.com/rpc
allowed_hosts: [mcp.example.com]
allowed_tools: [search_docs]
timeout_seconds: 30
max_response_bytes: 1000000
```

认证头只写环境变量名，真实密钥从进程环境读取（缺失时该 Server 请求
直接失败，不会发送空凭据）：

```yaml
auth_headers:
  Authorization: MARKET_MCP_TOKEN
auth_header_prefix: "Bearer "
```

本地 stdio Server 必须显式开启并声明命令允许清单；子进程只继承运行
必需环境变量，额外变量需显式透传：

```yaml
name: local-files
enabled: true
transport: stdio
command: [python, /opt/mcp/local_server.py]
stdio_allow_commands: [python]
stdio_env_passthrough: [LOCAL_MCP_ROOT]
allowed_tools: [read_file]
```

安全默认值：

- `enabled` 缺省为 `false`，必须显式开启；
- 默认只支持 HTTPS JSON-RPC，内网/回环地址默认拒绝；
- `allowed_hosts` 只支持精确主机名；
- `allowed_tools` 为空视为配置非法（默认拒绝全部工具）；
- `stdio` 默认禁止，必须 `allow_stdio: true` 且命令命中 `stdio_allow_commands`；
- 单个 Server 配置或连接失败只跳过该 Server，不影响其他条目和任务。

## 生效方式

- 下一个任务开始时按目录签名自动刷新；
- 显式刷新：`POST /api/tools/reload`（仅刷新校验结果，任务开始时注册）；
- 查看目录：`GET /api/tools/catalog`；
- 可选轮询监听：配置 `SO_AGENT_MCP_DIRECTORY_WATCH_SECONDS`（默认 0 关闭）。

工具会以 `<server-name>.<tool-name>` 注册，并仍需按任务签发 ToolGrant 才能被调用。
