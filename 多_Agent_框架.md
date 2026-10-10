# 多 Agent 框架结构蓝图（SO-Agent 受控执行运行时）

> **文档性质与状态标注约定**：本档保留立项时的原始总体设计思想，并逐章补充「实际演进」记录代码落地后的真实形态。能力一律使用四级标注：**【已实现】**（当前代码已有且经测试覆盖）、**【部分实现】**（核心在位、边界有限）、**【暂缓】**（设计理念保留、当前不做）、**【规划中】**（尚未实施）。英文对应文档 [SO_Agent_Execution_Runtime.md](./SO_Agent_Execution_Runtime.md) 与本档章节及能力状态一一对应。测试数与能力声明均为当前代码快照（1209 项测试），不构成永久承诺。

## 〇、实现状态总览

**已实现（1209 项测试，当前快照）**：四象限任务路由；薄 WorkflowEngine 与规划/执行/汇总三协调器；13 阶段影子 FSM；两阶段评审（`passed` 唯一放行 + fail closed + `suggestions` 非阻断）；授权 proposal 后置签发与全出口撤销；星型通信隔离；显式 `reuse_from` 复用；五层流式超时；PipelineGate 停止门；工具 timeout profile；SQLite schema 顺序增量迁移；FailureClassifier；失败指纹与去重；RecoveryPolicy；provider 禁替代；FORMAT 单次受限契约修复后 REPLAN；低信任 Agent 文本不可伪造系统失败；SQLite 持久化（任务 / Execution / PlanVersion / 事件日志）；路径隔离 SandboxRunner；MCP 协议层（protocol / server / client / adapter）；动态工具生成与 test_cases 语义验收；持久化调度守护进程。

**部分实现**：验证型沙箱（仅路径隔离，非安全边界）；人工兜底（仅 HumanEscalationHandler 接口与 `human_required` 状态，无人工处理端）；事件记录（结构化事件 + 本地日志文件，持久化审计链路未实施）。

**暂缓**：多维预算；上下文治理；生产容器沙箱（Docker / 远程隔离）。

**规划中**：统一 ModelGateway；权限执行级与参数级强化；Temporal PoC 或迁移；持久化审计；OpenTelemetry；动态 Agent 模板与能力白名单；Web 控制台与任务可视化；工具 TTL 缓存与调用计数持久化阈值（原 Redis 设想）；统计数据库分析表（原 MySQL 设想）。

## 一、设计目标

- 【已实现】主管 Agent 拥有全部工具的所有权，工具目录严格限定为代码 Agent、子 Agent 创建器、评审 Agent。
- 【已实现】主管 Agent 负责理解目标、拆分任务、授权工具、下发任务和组织最终答案，不直接访问文件系统或启动进程。
- 【已实现】子 Agent 默认不具备任何工具能力；主管 Agent 下发任务时为其指定本次任务允许使用的工具子集，未授权工具不可见且不可调用。
- 【已实现】子 Agent 之间禁止直接通信、共享会话或互相调用，所有输入、结果和反馈都必须经过主管 Agent。
- 【已实现】专用子 Agent 在运行时创建，用完释放，避免预置大量固定 Agent。
- 【已实现】评审覆盖任务拆解、授权范围和最终结果，并使用分层失败升级机制限制重试。
- 【已实现（真实次数由恢复策略决定）】每个子 Agent 最多尝试执行三次；原始子 Agent 失败后只能向主管 Agent 申请一次替代 Agent。
- 【已实现】替代 Agent 不具备再次申请替代 Agent 的能力；仍失败时触发任务拆分复查。
- 【已实现】主管 Agent 最多修改任务拆分三次，耗尽后进入人工兜底状态；首版只保留接口和状态，不实现人工处理系统。
- 【部分实现（仅路径隔离）】首版沙箱仅用于验证工具链：在独立目录写入代码并通过子进程运行，不提供真正的安全隔离。

## 二、总体分层

### 编排层

主管 Agent 位于最上层，接收用户目标并生成子任务计划。它独占工具目录的所有权，并负责决定每个子 Agent 能否使用工具以及可以使用哪些工具。主管 Agent 自身只注册以下三个 Agent 工具：

1. 代码 Agent 【已实现】
   - 拥有独立项目包、提示词文件和 API 配置文件。
   - 创建、修改和检查代码文件。
   - 必要时调用内部沙箱运行器验证代码。
   - 将新生成的工具写入专用的生成工具包，不写入其他 Agent 包。
   - 返回改动摘要、运行结果和错误信息。

2. 子 Agent 创建器 【已实现】
   - 拥有独立项目包、提示词文件和 API 配置文件。
   - 接收主管 Agent 签发的子任务定义和工具授权清单。
   - 根据角色、提示词、输入和授权范围动态构造专用 Agent。
   - 子 Agent 默认工具集合为空，仅挂载主管 Agent 明确批准的工具。
   - 调度无依赖子任务并发执行，按依赖关系串行推进其他任务。
   - 不向任何子 Agent 暴露其他 Agent 的引用、会话或消息通道。
   - 将每个子 Agent 的结构化结果分别返回主管 Agent，由主管 Agent 组织汇总。
   - 图中的汇总 Agent 属于按需创建的角色，只接收主管 Agent 提供的结果副本，不与其他子 Agent 通信，也不是第四个固定工具。

3. 评审 Agent 【已实现】
   - 拥有独立项目包、提示词文件和 API 配置文件。
   - 在执行前检查任务拆解是否完整、可执行且未偏离用户目标。
   - 检查每个子任务申请的工具是否必要，供主管 Agent 作出授权决定。
   - 在执行后检查结果正确性、完整性、证据和风险。
   - 输出通过或驳回、问题清单、修复要求及返工目标。
   - 图中的评估 Agent 与评审 Agent 合并为同一能力，不新增工具。

**实际演进（编排入口）**【已实现】：完整流水线之前增加四象限任务路由层（TaskRouter）——规则预判（纯数学表达式、明确问候）加一次轻量 LLM 分类，按「复杂度 × 时长」把任务分发到四条路径：直接回答（① 简单+短）、深度推理（② 复杂+短）、持久化任务（③ 简单+长，入库并由守护进程调度）、完整工作流（④ 复杂+长）。分类调用失败、解析失败或结果非法时，保守降级为完整工作流；`enable_router=False` 可整体关闭路由。简单任务的完整工作流内另有快速通道：子任务不超过 2 个、无依赖、全部成功且产出充分时跳过汇总与结果评审（`skip_result_review_for_simple=True`）。

### 控制层

控制层由普通 Python 代码实现，不作为模型可见工具：

- 状态机 【已实现，实际演进为影子视图】：约束任务只能按既定阶段流转；实际落地为 python-statemachine 声明式 TaskStateMachine——13 个高层阶段的影子图，作为权威 TaskStatus 的镜像，全部合法迁移显式声明，非法迁移被拒绝并留痕（详见第五章）。
- 调度器 【已实现】：处理依赖、并发上限、超时和取消（runtime/scheduler.py，SubtaskDAGScheduler；内置策略驱动重试与替换生命周期回调）。
- Agent 注册表 【已实现】：仅供主管 Agent 和控制层使用，记录动态 Agent 的身份、能力、状态和生命周期。
- 工具授权网关 【已实现，实际演进见第七章】：校验调用者、任务编号、工具名称、授权范围和有效期，拒绝一切未授权调用。
- 主管路由器 【已实现】：只允许主管 Agent 与各子 Agent 纵向通信，禁止子 Agent 横向传递消息。
- 重试控制器 【已实现，实际演进为确定性恢复内核】：限制返工次数，防止无限循环；实际由 FailureClassifier（失败分类）、失败指纹去重、RecoveryPolicy（决策矩阵）构成，逻辑全部在代码层，LLM 不可修改（详见第四章）。
- 事件记录器 【部分实现】：保存任务下发、授权、工具调用、输出、耗时和错误；当前为结构化事件 + 本地日志文件，跨任务持久化审计链路属规划中。
- 停止门（PipelineGate）【已实现】：deadline 到期、用户取消或人工终止后，在派发 / 重试 / 替换 / 重规划 / 采纳计划 / 签发授权等一切新动作入口拒绝放行，并隔离超时或取消后抵达的迟到结果。

### 基础设施层

这些属于内部实现服务，不计入主 Agent 的三个工具：

- 模型客户端 【已实现】：OpenAI Agents SDK Runner + AsyncOpenAI 客户端；分维度 HTTP 超时（connect / read / write / pool）可选注入。
- 文件读写适配器 【已实现】。
- 子进程运行器 【已实现】：asyncio 子进程执行生成工具与沙箱命令。
- 沙箱目录管理器 【已实现】：路径隔离 SandboxRunner。
- 配置和日志组件 【已实现】：全局 Settings（环境变量前缀 `SO_AGENT_`、`.env`、`--config` 文件三级覆盖）+ 各 Agent 独立 api_config.yaml + EventLogger。
- 持久化存储 【已实现】：SQLite TaskStore / ResultStore（含 schema 版本迁移机制）。
- MCP 基础设施 【已实现】：protocol（协议契约与标准错误码）、server（MCP Server 基类与 MCPToolServer）、client（进程内直连发现与调用）、adapter（Agent ↔ MCP Server ↔ SDK FunctionTool、生成工具 → MCP Server）四层。

**实际演进（模块对应）**：编排层与控制层按「薄编排器 + 三协调器 + 共享状态底座」组织——`workflow.py`（WorkflowEngine，仅主循环 / 人工兜底 / 输出组装）、`planning_coordinator.py`（拆解与计划评审）、`execution_coordinator.py`（调度执行与失败升级）、`aggregation_coordinator.py`（汇总与结果评审）、`state_machine.py`（FSM + WorkflowState + 领域异常 + 宽容解析）、`workflow_support.py`（WorkflowDeps 依赖容器、BaseCoordinator、PipelineGate、run_agent_streamed 流式执行）。协调器之间不互相引用，一律经共享 WorkflowState 交换状态。

## 三、完整运行流程

1. 接收任务 【已实现】
   - 创建任务编号。
   - 固化用户目标、约束和验收条件。

2. 主管 Agent 拆解与授权 【已实现】
   - 产生子任务列表。
   - 为每个子任务指定目的、输入、依赖、角色和完成条件。
   - 从三个工具中选择该子任务允许使用的工具子集；默认不授权任何工具。
   - 为每次下发生成不可由子 Agent 修改的授权记录。

3. 首轮评审 【已实现】
   - 主管 Agent 调用评审 Agent 检查拆解方案及工具授权是否必要、最小化。
   - 不通过时，根据反馈重新拆解或收紧授权。
   - 连续三次不通过则终止并返回明确原因。

4. 执行任务 【已实现】
   - 主管 Agent 将任务内容、上下文副本和允许工具清单封装后下发。
   - 编码任务可交给代码 Agent。
   - 研究、分析、汇总等任务可交给子 Agent 创建器，由其动态创建专用 Agent。
   - 子 Agent 只能调用本次任务明确授权的工具；未授权工具既不注入其工具集合，也会被授权网关拒绝。
   - 调度器可并发执行互不依赖的子任务，但各子 Agent 相互不可见、不可通信。
   - 每个执行结果只能返回主管 Agent，不能直接发送给其他子 Agent。

5. 汇总结果 【已实现】
   - 主管 Agent 收集各子 Agent 的独立结果。
   - 如需汇总角色，由主管 Agent 通过子 Agent 创建器创建，并只向其提供必要的结果副本和明确授权。
   - 汇总角色不获取其他子 Agent 的身份引用或通信能力。
   - 汇总结果必须保留每项结论对应的来源、执行状态和证据。

6. 最终评审 【已实现】
   - 评审 Agent 对照原始目标和验收条件检查结果。
   - 通过则由主管 Agent 输出最终结果。
   - 不通过则只返工被指出的任务，不重复执行已经通过的任务。

**实际演进**：
- 入口前置四象限路由（见第二章），仅「复杂+长」象限进入上述完整流程；
- 工具授权实际为 proposal 后置签发【已实现】：计划中只记录授权意图（proposal），计划评审通过后由代码层统一校验并签发 ToolGrant，防止未评审的授权直接生效；任务完成、替换、重规划、停止门触发等所有出口均撤销未完成授权（全出口撤销）；
- 重规划时主管可对新计划子任务显式声明 `reuse_from`（旧计划 subtask_id）【已实现】：是否放行由规划协调器安全校验器决定，代码绝不凭同名或指纹自动复用；复用判定权归主管 Agent。

### 子 Agent 失败升级流程

每一个任务拆分版本都按以下固定流程执行：

1. 原始子 Agent 执行 【已实现】
   - 单个子 Agent 最多执行三次。
   - 三次均未达到完成条件后，可向主管 Agent 发送一次 ReplacementRequest。
   - 该申请只能发送一次；重复申请由控制层直接拒绝。

2. 主管 Agent 处理替代申请 【已实现】
   - 主管 Agent 检查失败证据、已用工具和任务授权。
   - 同意后通过子 Agent 创建器生成一个新的替代 Agent，并重新下发同一子任务。
   - 主管 Agent 可调整替代 Agent 的提示词与授权工具，但不得改变原始主任务。
   - 【实际演进】首版中「主管检查失败证据后同意替换」由控制层自动批准（证据随 ReplacementRequest 全程留痕），不额外消耗模型调用；且 provider 类故障（模型服务商侧故障）禁止触发替换。

3. 替代 Agent 执行 【已实现】
   - 替代 Agent 同样最多执行三次。
   - 替代 Agent 的 can_request_replacement 固定为 false，不具备再次申请新 Agent 的能力。
   - 替代 Agent 仍失败时，只能向主管 Agent 返回 TaskDecompositionIssue，说明任务拆分可能存在问题及失败证据。

4. 主管 Agent 重新拆分 【已实现】
   - 主管 Agent 收到 TaskDecompositionIssue 后重新检查并修改任务拆分。
   - 主管 Agent 最多拥有三次拆分修改机会，每次生成新的计划版本并重新经过评审 Agent 检查。
   - 每个新计划版本重新创建原始子 Agent；该原始子 Agent 仍有一次替代申请资格，但其替代 Agent 仍无申请资格。

5. 人工兜底 【部分实现（接口与状态）】
   - 第三次任务拆分修改后仍无法完成，状态转为 human_required。
   - 控制层生成 HumanEscalationRequest，包含原始任务、所有计划版本、失败记录、工具调用证据和最后评审意见。
   - 首版只定义人工兜底接口、事件和状态，不实现人工工作台、通知或处理流程。

该机制同时限制单 Agent 重试、替代链深度和主管重规划次数，避免无限创建 Agent 或无限循环。

**实际演进（确定性失败恢复内核）**【已实现】：上述次数上限由恢复策略内核驱动，判定逻辑全部在代码层（runtime/failure.py 纯函数与数据矩阵，LLM 不可修改）：

- FailureClassifier 按异常类型与错误文本把失败分为 network / rate_limit / timeout / permission / budget / provider / format / executor_contract / quality 等类别；低信任来源（Agent 自报文本、工具与沙箱结果正文）不能伪造系统失败类别——恶意文本只会被归入 quality，结构性异常与执行器自身报告冲突时以执行器元数据为准；
- 每次失败生成稳定指纹，同指纹连续失败达到阈值（默认 2）即抑制重试，防止同因空转；去重状态与重试额度同步维护；
- RecoveryPolicy 按类别给出恢复动作：network / rate_limit 退避重试（基数 1 秒、上限 30 秒、抖动 0.2）；format 失败仅允许一次契约修复重试，修复后仍失败转入 REPLAN（重规划）且不触发替换；provider 类故障禁止替换；quality 失败在尝试额度内走业务重试 / 替换；
- 系统级网络连续失败（主管拆解阶段）耗尽独立网络重试预算后转入人工兜底，不消耗重规划额度。

## 四、核心状态模型

建议任务状态依次为：

- received：已接收。
- planning：正在拆分。
- plan_review：正在评审拆分结果。
- executing：正在执行子任务。
- replacement_requested：原始子 Agent 已申请替代执行者。
- replacement_executing：替代 Agent 正在执行同一子任务。
- decomposition_issue：替代 Agent 失败，等待主管 Agent 复查拆分。
- replanning：主管 Agent 正在修改任务拆分。
- aggregating：正在汇总结果。
- result_review：正在评审最终结果。
- completed：通过评审并完成。
- human_required：主管 Agent 三次修改任务后仍失败，等待人工介入。
- failed：遇到不可恢复的系统错误。
- cancelled：被用户或控制器取消。

每次状态迁移由控制层执行，Agent 只能提出结果或建议，不能自行跳过阶段。

**实际演进（影子 FSM）**【已实现】：TaskStateMachine 为 python-statemachine 声明式状态图，以 13 个高层阶段追踪流水线：idle → classifying → planning → reviewing_plan → granting_permissions → executing →（monitoring / replacing_agent / creating_tool）→ aggregating → reviewing_result →（reworking →）completed。它是权威 TaskStatus 的影子视图：每次真实状态迁移按映射表派发事件；failed / cancelled / human_required 三类终态没有对应阶段，影子图冻结在当前阶段供审计观察。全部合法迁移在图中显式声明，非法迁移被拒绝并留痕。替代状态链（replacement_requested → replacement_executing）由调度器在真实生命周期时点经受控回调实时映射，不再事后重建。

## 五、核心数据结构

### TaskRequest 【已实现】

- task_id
- objective
- constraints
- acceptance_criteria
- original_input

### SubtaskSpec 【已实现（含演进字段）】

- subtask_id
- plan_version
- title
- instructions
- role
- dependencies
- allowed_tools
- expected_output
- max_attempts
- reuse_from 【实际演进】：显式复用声明，值为旧计划 subtask_id；仅由主管在重规划时确信语义仍适用时填写；是声明元数据而非语义的一部分，不参与 spec_fingerprint；是否放行由规划协调器安全校验器决定。

### ToolGrant 【已实现】

- grant_id
- task_id
- agent_id
- allowed_tools
- issued_by
- issued_at
- expires_at
- revoked

### AgentRecord 【已实现】

- agent_id
- agent_type
- parent_task_id
- plan_version
- allowed_tools
- attempt_count
- can_request_replacement
- replacement_requested
- replacement_of
- status
- created_at
- expires_at

### ReplacementRequest 【已实现】

- request_id
- task_id
- subtask_id
- requester_agent_id
- failure_summary
- attempt_evidence
- requested_tools
- request_count

### TaskDecompositionIssue 【已实现】

- task_id
- subtask_id
- plan_version
- failed_agent_ids
- failure_reasons
- attempted_approaches
- recommendation

### ExecutionResult 【已实现】

- subtask_id
- success
- output
- evidence
- artifacts
- error
- duration

### Execution 【实际演进，已实现】

- 单次子任务执行记录，每次 attempt 生成一条（区别于 ExecutionResult 的最终态）：含尝试序号、失败类别（failure_type，存 FailureType.value 字符串）、恢复动作（recovery_action）、指纹与耗时等；随任务持久化到 SQLite，支撑审计与失败分析。

### PlanVersion 【实际演进，已实现】

- 计划版本记录：版本号、该版本 SubtaskSpec 列表、评审结论与来源（初次拆解 / 替换 / 重规划）；持久化保存，供重规划对照与 reuse_from 校验。

### ReviewDecision 【已实现（含演进字段）】

- stage（有意保持 str 类型，宽松接受 LLM 返回的各阶段值，调用方统一覆盖为 plan_review / result_review）
- passed（唯一放行字段；true 时 required_fixes 必须为空，false 时必须至少一条，由调用方校验，违反即 fail closed）
- issues
- suggestions 【实际演进】：非阻断改进建议，非空不影响放行、不构成拒绝理由
- required_fixes
- retry_target
- summary

### ClassificationResult 【实际演进，已实现】

- task_type（direct_answer / deep_reasoning / persistent_task / full_workflow）
- reasoning（一句话分类理由；分类失败时携带降级原因）

### GeneratedToolManifest 【已实现（含演进字段）】

- tool_id
- tool_name
- entry_file
- description
- input_schema
- output_schema
- created_by_task
- review_status
- version
- test_cases / expected_outputs 【实际演进】：验收测试用例数组与一一对应的期望返回值；注册前系统在沙箱中逐用例运行并语义比对，全部匹配才通过验收并注册。

### HumanEscalationRequest 【已实现】

- task_id
- original_request
- plan_history
- agent_failure_history
- tool_call_evidence
- final_review
- created_at

所有 Agent 间通信采用这些结构化对象，避免依赖自由文本猜测状态。

## 六、权限设计

- 【已实现】主管 Agent 拥有三个 Agent 工具的所有权、注册权、分配权和撤销权。
- 【已实现】子 Agent 默认无工具权限；主管 Agent 必须在每次任务下发时明确指定允许工具清单。
- 【已实现】工具权限按任务授予，不自动继承到后续任务，也不能由子 Agent 转授。
- 【已实现】创建子 Agent 时只把获批工具注入该 Agent 的工具集合；运行时授权网关再进行一次校验。
- 【已实现】子 Agent 申请额外工具时只能结束当前执行并向主管 Agent 返回申请原因，不能自行调用、发现或注册工具。
- 【已实现】原始子 Agent 连续三次失败后只能向主管 Agent 申请一次替代 Agent，该权限不可转让或重复使用。
- 【已实现】替代 Agent 在创建时固定移除替代申请能力，不能继续扩展替代链。
- 【已实现】子 Agent 之间没有消息通道、共享会话或彼此引用；替代申请、失败上报和结果交换都必须发送给主管 Agent。
- 【已实现】代码 Agent 只有在主管 Agent 授权后，才能使用其内部的项目目录读写能力和沙箱运行能力。
- 【已实现】代码 Agent 生成的新工具只保存为候选代码，不会自动注册、自动执行或自动授予任何子 Agent。
- 【已实现】候选工具通过评审后仍由主管 Agent 决定是否注册，以及在哪个具体任务中授权使用。
- 【已实现】评审 Agent 只有在主管 Agent 授权后运行，并保持只读，不修改产物。
- 【已实现】子 Agent 创建器只能按主管 Agent 签发的任务和授权创建 Agent，不能扩大权限、绕过注册表或突破并发上限。
- 【已实现】每个子任务和每次工具调用必须能追溯到主任务、授权记录和主管 Agent，不允许动态改变主任务。

**实际演进（授权生命周期）**【已实现】：

- proposal 后置签发：计划中记录授权意图（proposal），计划评审通过后由权限网关统一校验并签发 ToolGrant（含 expires_at）；未评审或评审未通过的授权意图不产生任何可用的授权凭证；
- 全出口撤销：任务完成 / 失败 / 取消、替代链触发、重规划采纳新计划、PipelineGate 停止等所有出口均撤销未完成授权（revoke_grants_for_agent / revoke_grants_for_task），撤销原因留痕；
- 授权粒度当前为工具名级；执行级（按调用上下文约束）与参数级（按入参约束）强化属规划中。

## 七、验证型沙箱

目录建议：

```text
sandbox/
  task_任务编号/
    run_运行编号/
      main.py
      stdout.txt
      stderr.txt
      metadata.json
```

运行机制：

- 使用 asyncio.create_subprocess_exec 创建独立 Python 进程，优先于线程。 【已实现】
- 使用当前虚拟环境的 Python 解释器。 【已实现】
- 将进程工作目录固定到本次运行目录。 【已实现】
- 设置运行超时并在超时后终止进程。 【已实现】
- 捕获标准输出、标准错误、退出码和耗时。 【已实现】
- 每次运行使用新目录，避免不同任务相互覆盖。 【已实现】
- 明确标注该方案只做路径隔离，不限制网络、CPU、内存、系统调用或目录逃逸，因此不能执行不可信代码。 【已实现（该限制为事实约束，见风险声明）】

**实际演进与暂缓**：原「沙箱网络白名单」设想（对生成工具的联网需求放行部分域名）经评估后确认：只要开网络口子就无法保证隔离，该方向已并入「生产容器沙箱」整体暂缓——容器级隔离（Docker / 远程执行环境）作为暂缓项保留设计理念，当前 SandboxRunner 仅适用于可信测试代码。

## 八、项目结构（实际）

```text
so-agent/
  pyproject.toml
  secrets.yaml              # 密钥注入源（不入库，main.py 启动加载到环境变量）
  src/
    so_agent/
      main.py               # CLI 入口（一次性任务 / --daemon 守护进程）
      config.py             # 全局 Settings（SO_AGENT_ 前缀环境变量 / .env / --config 覆盖）
      context.py            # ProjectContext（沙箱目录、项目名、配置）
      orchestrator.py       # 主管 Agent 定义与 OrchestratorOutput
      workflow.py           # WorkflowEngine 薄编排器（主循环 / 人工兜底 / Facade 兼容代理）
      workflow_support.py   # WorkflowDeps / BaseCoordinator / PipelineGate / run_agent_streamed
      state_machine.py      # TaskStateMachine（13 阶段影子 FSM）/ WorkflowState / 领域异常
      planning_coordinator.py
      execution_coordinator.py
      aggregation_coordinator.py
      router.py             # TaskRouter 四象限路由
      paths/                # direct_answer / deep_reasoning / persistent_task
      models.py             # 全部结构化输入输出模型
      fingerprint.py        # spec 指纹
      json_utils.py         # JSON 宽容提取
      persistence.py        # SQLite TaskStore
      result_store.py       # ResultStore（内存 / SQLite）与结果指纹存储
      schema_migrations.py  # schema 版本机制与顺序增量迁移
      task_scheduler.py     # PeriodicTaskPoller 守护进程轮询
      api_config.yaml(.example)
      mcp/                  # protocol / server / client / adapter
      runtime/
        config_loader.py    # api_config.yaml 加载与分维度 HTTP 超时解析
        events.py           # EventLogger 结构化事件
        failure.py          # FailureClassifier / 指纹 / RecoveryPolicy
        permissions.py      # PermissionGateway（proposal / grant / revoke）
        registry.py         # AgentRegistry
        sandbox.py          # SandboxRunner（路径隔离）
        scheduler.py        # SubtaskDAGScheduler（策略驱动重试）
        supervisor_router.py # 星型通信强制
        tool_registry.py    # 工具目录（三 Agent 工具 + 已注册生成工具）
        generated_tool_runner.py # 生成工具子进程执行与语义验收
      tool_packages/
        code_agent/         # agent.py + prompt.md + api_config.yaml
        review_agent/       # agent.py + prompt.md + api_config.yaml
        subagent_creator/   # agent.py + prompt.md + api_config.yaml
        generated_tools/    # 生成工具仓库（每工具独立子目录 + registry.json）
  sandbox/                  # 运行时工作区（任务 / 沙箱运行目录 / 任务库）
  tests/
    unit/  integration/  end_to_end/  baseline/
```

配置安全规则 【已实现】：

- API 配置文件保存 provider、model、base_url、timeout、retry（及可选分维度 HTTP 超时）和密钥环境变量名（api_key_env）。
- API 密钥不写入仓库：经 api_key_env 指定的环境变量或根目录 secrets.yaml 注入；api_key 直接写值仅为兼容保留且不推荐。
- 三个 Agent 可使用不同模型或服务地址，但均由统一 config_loader.py 解析和校验；api_key 字段序列化时自动排除，绝不写入日志。

核心文件职责：

- orchestrator.py：定义主管 Agent，持有且仅持有三个 Agent 工具入口。
- workflow.py：薄编排器——主循环、状态分派、人工兜底与输出组装；两阶段评审与失败升级分属各协调器。
- planning_coordinator.py / execution_coordinator.py / aggregation_coordinator.py：规划 / 执行 / 汇总三协调器。
- state_machine.py：13 阶段影子 FSM、共享可变状态 WorkflowState、工作流领域异常与结构化输出宽容解析。
- workflow_support.py：协调器依赖容器、停止门 PipelineGate 与五层流式超时执行器 run_agent_streamed。
- router.py：四象限分类与路由分发。
- persistence.py / result_store.py / schema_migrations.py：SQLite 持久化与版本迁移。
- runtime/failure.py：失败分类、指纹与恢复策略（纯代码决策）。
- runtime/permissions.py：授权 proposal、签发、校验与撤销。
- runtime/scheduler.py：依赖图调度、隔离并发、恢复策略驱动重试与替换生命周期。
- runtime/sandbox.py：路径隔离子进程执行。
- runtime/config_loader.py：各级 api_config.yaml 加载、密钥解析与 HTTP 超时解析。
- 人工兜底仅在 workflow.py / workflow_support.py 中保留 HumanEscalationHandler 接口和 human_required 状态，不实现外部人工系统。

## 九、SDK 使用边界

- 采用 Python OpenAI Agents SDK。 【已实现】
- 主管 Agent 通过 Agent.as_tool 持有三个 Agent 工具的唯一注册入口。 【已实现】
- 三个固定 Agent 分别从自己的包中加载提示词和 API 配置，避免职责与模型参数相互耦合。 【已实现】
- 动态角色由子 Agent 创建器在运行时构造 Agent，默认 tools 为空，只能注入 ToolGrant 中获批的工具。 【已实现】
- 代码 Agent 生成的工具先写入 generated_tools，评审通过且主管 Agent 注册后才可进入授权候选集合。 【已实现】
- 不为子 Agent 配置 handoff，也不共享 session，避免建立任何横向通信路径。 【已实现】
- 评审 Agent 使用 Pydantic 模型输出 ReviewDecision。 【已实现】
- Runner 负责单次 Agent 执行，外层 workflow 负责授权、路由和业务重试，二者职责分离。 【已实现】
- 每次运行显式设置最大轮次、模型超时和工具超时。 【已实现】
- 在项目独立虚拟环境中安装依赖，避免直接升级当前全局环境中的 openai 包。 【已实现】

**实际演进（流式执行与超时）**【已实现】：

- Agent 调用统一经 run_agent_streamed 流式执行，五层窗口：first_event_timeout（首事件）→ model_idle_timeout（模型空闲，仅 SDK 原生事件可续时）→ tool_execution_timeout（普通工具窗口）→ agent_total_timeout / supervisor_total_timeout（调用总预算）→ max_pipeline_timeout（流水线 deadline，到期优雅降级）；
- 嵌套子 Agent 与长时代码 Agent 使用独立工具窗口（默认派生为 agent_total_timeout + 10 秒清理宽限），总预算仍在运行时被 pipeline 剩余额度截断；
- HTTP 客户端超时与流式阶段超时是两回事：connect / read / write / pool 分维度超时只负责单次 I/O 快速失败，总时长由上层窗口兜底；
- CLI 启动时默认禁用 SDK trace 导出（OPENAI_AGENTS_DISABLE_TRACING=1），消除无 Tracing key 的警告。

## 十、配置（实际默认值）

以下为当前代码中的真实默认值（config.py），均可经环境变量（前缀 `SO_AGENT_`）、`.env` 或 `--config` 文件覆盖，不写死在提示词中。这些是运行时间上限与治理额度，**不构成已实现的多维预算体系**（预算与上下文治理暂缓）。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| max_dynamic_agents | 8 | 动态子 Agent 数量上限 |
| max_agent_attempts | 3 | 单 Agent 尝试上限（真实次数由恢复策略决定） |
| max_replacement_requests | 1 | 替代申请上限（替代链深度 1） |
| max_replan_attempts | 3 | 主管拆分修订上限 |
| max_concurrency | 3 | 并发子任务上限 |
| max_model_turns | 8 | 单次 Agent 运行最大模型回合数 |
| model_timeout | 45 秒 | 单次模型调用超时（HTTP 层） |
| first_event_timeout | 180 秒 | 流式首事件窗口 |
| model_idle_timeout | 90 秒 | 模型阶段空闲窗口 |
| tool_execution_timeout | 180 秒 | 普通工具执行窗口 |
| subagent_tool_execution_timeout | 610 秒 | 嵌套子 Agent 工具窗口（派生 = agent_total + 10） |
| code_agent_tool_execution_timeout | 610 秒 | 代码 Agent 工具窗口（派生 = agent_total + 10） |
| agent_total_timeout | 600 秒 | 单次 Agent 调用总预算 |
| supervisor_total_timeout | 900 秒 | 主管单次调用总预算（保底不短于嵌套工具窗口） |
| max_pipeline_timeout | 2400 秒 | 流水线总 deadline（到期优雅降级） |
| sandbox_timeout | 30 秒 | 沙箱单次命令超时 |
| sandbox_output_limit | 1 MiB | 沙箱输出上限 |
| classifier_max_tokens | 512 | 路由分类调用最大输出 token |
| enable_router | true | 四象限路由开关（False 走完整工作流） |
| skip_result_review_for_simple | true | 简单任务快速通道（跳过汇总与结果评审） |
| recovery_backoff_base_seconds | 1.0 | network/rate_limit 退避基数 |
| recovery_backoff_cap_seconds | 30.0 | 单次退避上限 |
| recovery_backoff_jitter_ratio | 0.2 | 退避抖动比例 |
| recovery_retry_suppress_threshold | 2 | 同指纹连续失败抑制阈值 |
| recovery_repair_contract_max_attempts | 1 | FORMAT 契约修复重试上限 |
| task_store_path | sandbox/task_store.db | SQLite 任务存储路径 |
| scheduler_poll_interval | 60 秒 | 守护进程轮询间隔 |

子 Agent 默认工具数为 0；工具授权有效期仅限单次子任务执行；子 Agent 横向通信始终禁用。旧超时字段（max_idle_timeout / max_supervisor_call_timeout）仅为兼容别名，显式设置时映射到新字段。

## 十一、测试计划

- 【已实现】单元测试：状态迁移、依赖调度、授权签发与撤销、注册表上限、沙箱超时和各级次数上限。
- 【已实现】包结构测试：确认工具类包固定包含三个 Agent 包和一个生成工具包，三个 Agent 包均具有提示词与 API 配置。
- 【已实现】配置测试：确认各级 API 配置可独立加载、缺失字段能被拒绝且仓库内不存在明文密钥。
- 【已实现】主体配置测试：确认主管 Agent 工具集合严格等于三个指定工具，子 Agent 默认工具集合为空。
- 【已实现】生成工具测试：确认代码 Agent 只能写入 generated_tools，候选工具未经评审和主管注册时不可调用；语义验收逐用例比对。
- 【已实现】权限测试：确认子 Agent 只能调用本任务获批工具，无法转授权限，过期或越权调用必定失败；授权 proposal 后置签发与全出口撤销有专项覆盖。
- 【已实现】通信隔离测试：确认子 Agent 无法获取彼此引用、共享会话或直接发送消息。
- 【已实现】替代流程测试：确认原始子 Agent 三次失败后只能申请一次，替代 Agent 没有申请能力且失败后只能报告任务拆分问题；provider 禁替代有专项覆盖。
- 【已实现】重规划测试：确认主管 Agent 最多修改拆分三次，第四次不会继续运行而是生成 HumanEscalationRequest；显式 reuse_from 复用校验有专项覆盖。
- 【已实现】人工兜底测试：确认只产生 human_required 状态和完整交接数据，不调用未实现的人工系统。
- 【已实现】失败恢复测试：失败分类矩阵、指纹去重、退避注入、FORMAT 修复后 REPLAN、低信任文本不可伪造系统失败。
- 【已实现】超时与停止门测试：五层流式超时、停止门拒绝新动作、迟到结果隔离。
- 【已实现】schema 迁移测试：顺序增量迁移、幂等、未知高版本 fail closed。
- 【已实现】模拟集成测试：使用假模型验证拆解、授权、动态创建、替代、重规划、汇总、驳回和人工兜底流程，不消耗真实 API。
- 【已实现】沙箱测试：验证代码写入、子进程执行、输出捕获、失败退出和超时终止。
- 【部分实现】真实端到端测试：在提供 API 密钥后执行一个完整任务，并确认最终结果经过两阶段评审；此类测试需真实密钥，不在默认测试集内。

**现状**：当前测试集共 **1209 项（当前快照）** 全部可离线通过；该数字随代码演进变化，不作为永久能力保证。

## 十二、分阶段落地（进度对照）

### 最小可运行版本 【已实现】

- 建立项目、虚拟环境和四个工具项目包。 ✅
- 为三个固定 Agent 分别建立提示词文件和 API 配置文件。 ✅
- 定义结构化模型与运行上下文。 ✅
- 实现三个固定 Agent 工具及生成工具仓库。 ✅
- 实现动态 Agent 注册表。 ✅
- 实现验证型沙箱。 ✅（路径隔离）
- 实现两阶段评审、子 Agent 三次尝试、一次替代申请和主管 Agent 三次重规划。 ✅
- 定义人工兜底接口与状态，不实现人工处理端。 ✅
- 提供命令行入口和模拟测试。 ✅

### 稳定性增强 【部分实现】

- 持久化事件日志、任务恢复和取消机制 【已实现】：EventLogger 结构化事件、SQLite 任务库与守护进程、PipelineGate 与协作式取消。跨任务持久化审计链路仍属规划中。
- token、耗时和并发预算 【部分实现 / 部分暂缓】：并发上限与全套时间上限已实现，基本 token usage 有记录；多维预算体系暂缓。
- 动态 Agent 模板与能力白名单 【规划中】。

### 生产化增强 【规划中 / 部分暂缓】

- 将验证型沙箱替换为 Docker 或远程隔离执行环境 【暂缓（设计保留）】。
- 可观测性（OpenTelemetry）、持久化审计、多租户隔离 【规划中】；密钥管理 【已实现（secrets.yaml 集中注入 + 环境变量）】。
- Web 控制台和任务可视化 【规划中】。

## 十三、关键假设与风险

- 【存档：立项时假设，已不适用】工作区当前为空，可按全新项目设计，不考虑旧代码兼容。
- 【已实现】项目使用独立 Python 虚拟环境。
- 【风险，必须知悉】代码运行目录并非安全沙箱，只适用于可信测试代码；不能安全运行不可信代码。
- 【已实现】动态 Agent 不会自动成为主管 Agent 的新工具，而是在子 Agent 创建器内部完成创建和执行。
- 【已实现】generated_tools 中的代码只是候选工具，未经评审和主管 Agent 注册不会进入可调用范围。
- 【已实现】各级 API 配置相互独立，但密钥统一只从环境变量（或 secrets.yaml 注入）读取。
- 【已实现】主管 Agent 是唯一的工具所有者和消息中枢，子 Agent 不能自行扩大权限或相互通信。
- 【已实现】每个计划版本最多包含一层替代 Agent，不允许替代 Agent 继续申请替代者。
- 【已实现】主管 Agent 最多修改任务拆分三次，之后必须进入人工兜底状态而不是继续自动重试。
- 【部分实现】人工兜底仅为逻辑占位，不包含界面、通知、工单或人工回写能力。
- 【已实现】模型输出即使使用结构化类型仍可能失败，控制层必须捕获解析错误（宽容解析 + fail closed）。
- 【已实现】模型调用存在成本和不确定性，测试应优先使用假模型。
- 【风险】本项目为实验 / 研究状态，治理与恢复机制未经生产负载检验。
- 【风险】底层流式取消可能延迟：超时 / 取消收尾依赖底层 SDK 取消行为，关闭迭代器有有限宽限；迟到结果由停止门隔离，但不保证资源即时释放。
- 【暂缓】多维预算与上下文治理暂缓期间，长任务成本与上下文控制依赖时间上限与有限重试额度。
