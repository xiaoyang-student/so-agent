<div align="right">
  <a href="English_README.md">
    <img src="https://img.shields.io/badge/lang-English-blue.svg" alt="English">
  </a>
</div>

# SO-Agent

受控的多 Agent 执行运行时：基于 [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) 构建的两层星型多 Agent 编排框架。核心设计原则是 **LLM 负责思考，代码负责决策**——LLM 负责理解目标、拆解任务、生成指令与分析内容；状态流转、权限签发、重试控制与执行调度等一切决策均由确定性代码层做出。

> **项目状态：实验性 / 研究用途。** 不适用于生产环境，接口与行为可能随时变化。测试规模与能力声明均为当前代码快照，不构成永久承诺。

- 中文架构文档：[多_Agent_框架.md](./多_Agent_框架.md)
- English architecture document: [SO_Agent_Execution_Runtime.md](./SO_Agent_Execution_Runtime.md)


## 架构概览

<!-- Mermaid 流程图 -->
graph TD
    %% --- 样式定义 ---
    classDef purple fill:#e6e6fa,stroke:#333,stroke-width:1px;
    classDef green fill:#d0f0c0,stroke:#333,stroke-width:1px;
    classDef orange fill:#ffdab9,stroke:#333,stroke-width:1px;
    classDef yellow fill:#fffacd,stroke:#333,stroke-width:1px;
    
    %% 强制子图背景色（GitHub Mermaid 支持有限，主要靠节点颜色区分，这里尝试设置子图样式）
    %% 注意：GitHub 渲染器可能不完全支持 subgraph 的背景色设置，但逻辑结构会保留。

    %% ================= 阶段 1 =================
    subgraph Phase1 ["阶段1: 任务路由"]
        A1[用户输入]:::purple
        D1{规则预判/轻量LLM分类}:::green
        
        R1[直接回答]:::purple
        R2[深度推理]:::purple
        R3[持久化任务]:::purple
        R4[完整工作流]:::purple

        A1 --> D1
        D1 -- 低复杂度/短时长 --> R1
        D1 -- 高复杂度/短时长 --> R2
        D1 -- 低复杂度/长时长 --> R3
        D1 -- 高复杂度/长时长 或<br/>分类失败 --> R4
    end

    %% ================= 阶段 2 =================
    subgraph Phase2 ["阶段2: 规划与评审"]
        S1((状态: Planning)):::yellow
        P1[Supervisor 拆解任务 & 生成 Proposal]:::orange
        S2((状态: Plan Review)):::yellow
        D2{Review Agent 评审}:::orange
        G1[控制层: 签发权限 Grant]:::green
        D3{"重规划次数 < 3?"}:::purple
        S_Human((状态: Human Required)):::yellow

        R4 --> S1
        S1 --> P1
        P1 --> S2
        S2 --> D2
        D2 -- passed=true --> G1
        D2 -- passed=false --> D3
        D3 -- 是 --> P1
        D3 -- 否 --> S_Human
    end

    %% ================= 阶段 3 & 4 =================
    subgraph Phase3 ["阶段3 & 4: 执行与失败恢复"]
        S3((状态: Executing)):::yellow
        P2[Supervisor 下发任务给 Worker]:::purple
        D4{Worker 是否需要新工具?}:::purple
        
        P3[Code Agent 生成代码 & Test Cases]:::orange
        P4[Sandbox 沙箱验证]:::green
        
        P5[注册工具 & 注入 Worker]:::purple
        P6[Worker 执行任务]:::orange
        D5{执行结果}:::purple
        
        R5[返回结果给 Supervisor]:::purple
        P7[FailureClassifier 分类 & 指纹去重]:::green
        D6{RecoveryPolicy 决策}:::green
        S_Replan((状态: Replanning)):::yellow

        G1 --> S3
        S3 --> P2
        P2 --> D4
        
        D4 -- 是 --> P3
        P3 -- 测试失败 --> P3
        P3 -- 测试通过 --> P4
        P4 --> P5
        
        D4 -- 否 --> P5
        P5 --> P6
        P6 --> D5
        
        D5 -- 成功 --> R5
        D5 -- 失败 --> P7
        P7 --> D6
        
        D6 -- 重试/替代深度1 --> P6
        D6 -- 重规划 --> S_Replan
        D6 -- 耗尽资源 --> S_Human
    end

    %% ================= 阶段 5 & 6 =================
    subgraph Phase5 ["阶段5 & 6: 汇总与清理"]
        S4((状态: Aggregating)):::yellow
        P8[Supervisor 汇总结果]:::orange
        S5((状态: Result Review)):::yellow
        D7{Review Agent 评审}:::orange
        D8{允许修复?}:::purple
        
        S_Completed((状态: Completed)):::yellow
        P9[Worker 修复结果]:::purple
        G2[控制层: 撤销权限 Revoke]:::green
        P10[持久化到 SQLite]:::green
        R6[返回用户]:::purple

        R5 --> S4
        S4 --> P8
        P8 --> S5
        S5 --> D7
        
        D7 -- passed=true --> S_Completed
        D7 -- passed=false --> D8
        
        D8 -- 是 --> P9
        P9 --> R5
        D8 -- 否 --> S_Replan
        
        S_Completed --> G2
        G2 --> P10
        P10 --> R6
    end

    %% ================= 跨阶段连接 =================
    S_Replan ==> P1
    S_Human -.-> S_Replan

    %% 子图样式调整 (仅部分渲染器支持)
    style Phase1 fill:#2b2b2b,stroke:#fff,color:#fff
    style Phase2 fill:#2b2b2b,stroke:#fff,color:#fff
    style Phase3 fill:#2b2b2b,stroke:#fff,color:#fff
    style Phase5 fill:#2b2b2b,stroke:#fff,color:#fff

## 核心特性（已实现）

以下能力均已在当前代码中实现，并经 **1209 项测试（当前快照）** 覆盖验证：

- **四象限任务路由**：规则预判 + 一次轻量 LLM 分类，按「复杂度 × 时长」将任务分发到直接回答 / 深度推理 / 持久化任务 / 完整工作流四条路径；分类失败时保守降级为完整工作流。
- **两层星型架构**：主管 Agent 独占且仅持有三个工具入口（code_agent、subagent_creator、review_agent）；子 Agent 之间无消息通道、无共享会话，一切通信经主管 Agent 中转。
- **薄编排器 + 三协调器**：WorkflowEngine 只保留主循环与人工兜底，规划（拆解与计划评审）、执行（调度与失败升级）、汇总（聚合与结果评审）分属三个协调器，经共享状态交换信息。
- **13 阶段影子 FSM**：python-statemachine 声明式状态图作为权威任务状态的影子视图，所有合法迁移显式声明，非法迁移被拒绝并留痕。
- **两阶段评审**：计划评审与结果评审均产出结构化 ReviewDecision；`passed` 是唯一放行字段，fail closed——契约无效或解析失败一律保守判定未通过；`suggestions` 为非阻断建议，不构成拒绝理由。
- **授权 proposal 后置签发与全出口撤销**：工具授权由代码层在计划评审通过后统一签发（proposal → grant），任务结束、替换、重规划等所有出口均撤销授权。
- **确定性失败恢复**：FailureClassifier 对失败分类（低信任 Agent 文本不能伪造系统失败类别）、失败指纹去重（同因连续失败达阈值即抑制重试）、RecoveryPolicy 按类别给出重试 / 修复 / 替换 / 重规划决策；provider 类故障禁止替换，输出格式（FORMAT）失败仅允许一次契约修复后转入重规划。
- **替代链深度 1 与有限重规划**：单 Agent 最多尝试 3 次、原始子 Agent 只能申请一次替代、替代 Agent 无再申请权、主管最多 3 次拆分修订（支持显式 `reuse_from` 复用旧结果），耗尽后进入 `human_required` 人工兜底状态。
- **五层流式超时与停止门**：首事件 / 模型空闲 / 工具执行 / 单 Agent 总预算 / 流水线总 deadline 五层窗口；PipelineGate 停止门在 deadline 到期或取消后拒绝一切新动作并隔离迟到结果。
- **工具 timeout profile**：普通工具、嵌套子 Agent、长时代码 Agent 使用独立的工具阶段等待窗口，避免长任务被通用窗口误杀。
- **动态工具生成与 MCP 层**：code_agent 生成工具须提供 test_cases 自证正确，注册前在沙箱中逐用例语义验收；协议层提供 MCP Server / Client / 双向适配器。
- **持久化与 schema 迁移**：SQLite 存储长事务任务、执行记录（Execution）与计划版本（PlanVersion）；版本化顺序增量迁移，幂等、事务化、未知高版本 fail closed。

## 安装

要求 Python 3.10+。以下命令均适用于 Windows PowerShell（不使用 `&&` 连接符）。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## 密钥配置

密钥不写入仓库。两种方式（任选其一）：

1. **环境变量**：为各 Agent 的 `api_key_env` 所指环境变量赋值，例如：

   ```powershell
   $env:SO_AGENT_API_KEY = "sk-your-key"
   ```

2. **secrets.yaml**（项目根目录，已被版本控制忽略）：`main.py` 启动时自动加载其中的键值对到环境变量。

各 Agent 的连接配置位于 `src/so_agent/api_config.yaml` 与 `src/so_agent/tool_packages/<agent>/api_config.yaml`。推荐写法是声明 `api_key_env`（环境变量名）；`api_key` 直接写值仅为兼容保留且不推荐。示例见各级 `api_config.yaml.example`。

## 快速开始

```powershell
# 一次性任务（任务目标也可从标准输入管道读取）
.\.venv\Scripts\python.exe -m so_agent.main "为目标目录实现一个 CLI 统计工具" --constraints "仅使用标准库" --acceptance-criteria "提供可运行脚本并自我验证"

# 持久化任务调度守护进程（后台轮询并执行到期任务，无需任务目标）
.\.venv\Scripts\python.exe -m so_agent.main --daemon
```

说明：

- `--config <文件>` 可用 YAML / JSON 覆盖 `Settings` 默认字段（如 `max_concurrency`、`model_timeout`）；
- 任务终态为 `completed` 时退出码为 0，其余终态（`human_required` / `failed` / `cancelled`）为 1，参数或配置错误为 2；
- 标准输出仅打印结构化 JSON 结果，运行状态摘要打印到标准错误。

## 运行测试

```powershell
.\.venv\Scripts\python.exe -m pytest
```

当前收集 **1209 项测试（当前快照）**，全部使用假模型，不消耗真实 API；真实端到端测试需提供密钥后单独执行。

## 主要默认配置（当前代码实测）

以下均为运行时间上限与治理参数，可经环境变量（前缀 `SO_AGENT_`）、`.env` 或 `--config` 文件覆盖。它们**不是**已实现的多维预算体系（预算与上下文治理当前暂缓，见下文）。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `max_model_turns` | 8 | 单次 Agent 运行最大模型回合数 |
| `model_timeout` | 45 秒 | 单次模型调用超时（HTTP 层） |
| `first_event_timeout` | 180 秒 | 流式首事件窗口 |
| `model_idle_timeout` | 90 秒 | 模型阶段空闲窗口 |
| `tool_execution_timeout` | 180 秒 | 普通工具执行窗口 |
| `subagent_tool_execution_timeout` | 610 秒 | 嵌套子 Agent 工具窗口（默认派生 = Agent 总预算 + 10 秒） |
| `code_agent_tool_execution_timeout` | 610 秒 | 代码 Agent 工具窗口（同上派生） |
| `agent_total_timeout` | 600 秒 | 单次 Agent 调用总预算 |
| `supervisor_total_timeout` | 900 秒 | 主管单次调用总预算 |
| `max_pipeline_timeout` | 2400 秒 | 流水线总 deadline（到期优雅降级） |
| `max_dynamic_agents` | 8 | 动态子 Agent 数量上限 |
| `max_concurrency` | 3 | 并发执行子任务数上限 |
| `max_agent_attempts` | 3 | 单 Agent 尝试次数上限（真实次数由恢复策略决定） |
| `max_replacement_requests` | 1 | 替代申请次数上限（替代链深度 1） |
| `max_replan_attempts` | 3 | 主管拆分修订次数上限 |
| `sandbox_timeout` | 30 秒 | 沙箱单次命令超时 |
| `sandbox_output_limit` | 1 MiB | 沙箱输出上限 |
| `scheduler_poll_interval` | 60 秒 | 守护进程轮询间隔 |
| `task_store_path` | `sandbox/task_store.db` | SQLite 任务存储路径 |

## 能力状态与路线图

严格区分实现状态，避免把愿景写成现有能力。

### 已实现

四象限路由、薄 WorkflowEngine 与三协调器、13 阶段影子 FSM、两阶段评审（`passed` 唯一放行 + fail closed）、授权 proposal 后置签发与全出口撤销、星型通信隔离、显式 `reuse_from` 复用、五层流式超时、PipelineGate、工具 timeout profile、schema migration、FailureClassifier、失败指纹与去重、RecoveryPolicy、provider 禁替代、FORMAT 单次受限契约修复后 REPLAN、低信任 Agent 文本不可伪造系统失败、SQLite 持久化（任务 / Execution / PlanVersion）、路径隔离 SandboxRunner、MCP 协议层、动态工具语义验收注册。

### 暂缓（设计理念保留）

- **多维预算**：token / 成本 / 时间的多维度预算体系。现存的各类时间上限（见默认配置表）与基本 token usage 记录是单一维度约束，不等于多维预算。
- **上下文治理**：长任务的上下文压缩、裁剪与配额管理。
- **生产容器沙箱**：现行 SandboxRunner 仅做路径隔离（见风险声明），容器级隔离（原 Docker / 远程隔离执行设想）暂缓。

### 规划中（未实现）

- **统一 ModelGateway**：模型接入、配额与故障转移的统一网关层。
- **权限执行级与参数级强化**：现有授权为工具名粒度；按参数细粒度约束的强化尚未实施。
- **Temporal PoC 或迁移**：以 Temporal 替代进程内编排的可行性验证。
- **持久化审计**：跨任务的完整审计链路存储与查询。
- **OpenTelemetry**：分布式追踪与可观测性接入。
- 其他保留的未来方向：工具 TTL 缓存与调用计数驱动的持久化阈值（原 Redis 设想，现行以显式持久化调度替代）、统计数据库分析表（原 MySQL 设想，现行以 SQLite 执行记录与本地日志替代）、动态 Agent 模板与能力白名单、Web 控制台与任务可视化。

## 风险与限制

- **实验或研究状态**：本项目的治理与恢复机制面向研究与验证，未经生产负载检验。
- **底层流式取消可能延迟**：超时或取消收尾依赖底层 SDK 的取消行为；关闭迭代器有有限宽限，极端情况下底层取消可能延迟，迟到结果由停止门隔离但不保证资源即时释放。
- **当前沙箱不是安全边界**：SandboxRunner 只做工作目录路径隔离，不限制网络、CPU、内存、系统调用或目录逃逸，**不能安全运行不可信代码**。
- **人工兜底只有接口与状态**：`human_required` 终态会产出完整的 HumanEscalationRequest 交接数据，但不包含人工工作台、通知或工单回写系统。
- **预算与上下文暂缓**：多维预算与上下文治理暂缓期间，长任务的成本与上下文控制依赖上述时间上限与有限重试额度。

## 文档

- [多_Agent_框架.md](./多_Agent_框架.md)：中文架构文档——原始总体设计思想与实际演进、分层、状态模型、数据结构、权限、配置与测试计划。
- [SO_Agent_Execution_Runtime.md](./SO_Agent_Execution_Runtime.md)：English architecture document（与中文版章节和能力状态一一对应）。
- 配置示例：[src/so_agent/api_config.yaml.example](./src/so_agent/api_config.yaml.example) 及 `src/so_agent/tool_packages/<agent>/api_config.yaml.example`。
