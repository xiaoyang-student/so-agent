# 多 Agent 框架结构蓝图

## 一、设计目标

- 主管 Agent 拥有全部工具的所有权，工具目录严格限定为代码 Agent、子 Agent 创建器、评审 Agent。
- 主管 Agent 负责理解目标、拆分任务、授权工具、下发任务和组织最终答案，不直接访问文件系统或启动进程。
- 子 Agent 默认不具备任何工具能力；主管 Agent 下发任务时为其指定本次任务允许使用的工具子集，未授权工具不可见且不可调用。
- 子 Agent 之间禁止直接通信、共享会话或互相调用，所有输入、结果和反馈都必须经过主管 Agent。
- 专用子 Agent 在运行时创建，用完释放，避免预置大量固定 Agent。
- 评审覆盖任务拆解、授权范围和最终结果，并使用分层失败升级机制限制重试。
- 每个子 Agent最多尝试执行三次；原始子 Agent失败后只能向主管 Agent申请一次替代 Agent。
- 替代 Agent不具备再次申请替代 Agent的能力；仍失败时触发任务拆分复查。
- 主管 Agent最多修改任务拆分三次，耗尽后进入人工兜底状态；首版只保留接口和状态，不实现人工处理系统。
- 首版沙箱仅用于验证工具链：在独立目录写入代码并通过子进程运行，不提供真正的安全隔离。

## 二、总体分层

### 编排层

主管 Agent 位于最上层，接收用户目标并生成子任务计划。它独占工具目录的所有权，并负责决定每个子 Agent 能否使用工具以及可以使用哪些工具。主管 Agent 自身只注册以下三个 Agent 工具：

1. 代码 Agent
   - 拥有独立项目包、提示词文件和 API 配置文件。
   - 创建、修改和检查代码文件。
   - 必要时调用内部沙箱运行器验证代码。
   - 将新生成的工具写入专用的生成工具包，不写入其他 Agent 包。
   - 返回改动摘要、运行结果和错误信息。

2. 子 Agent 创建器
   - 拥有独立项目包、提示词文件和 API 配置文件。
   - 接收主管 Agent 签发的子任务定义和工具授权清单。
   - 根据角色、提示词、输入和授权范围动态构造专用 Agent。
   - 子 Agent 默认工具集合为空，仅挂载主管 Agent 明确批准的工具。
   - 调度无依赖子任务并发执行，按依赖关系串行推进其他任务。
   - 不向任何子 Agent 暴露其他 Agent 的引用、会话或消息通道。
   - 将每个子 Agent 的结构化结果分别返回主管 Agent，由主管 Agent 组织汇总。
   - 图中的汇总 Agent属于按需创建的角色，只接收主管 Agent 提供的结果副本，不与其他子 Agent 通信，也不是第四个固定工具。

3. 评审 Agent
   - 拥有独立项目包、提示词文件和 API 配置文件。
   - 在执行前检查任务拆分是否完整、可执行且未偏离用户目标。
   - 检查每个子任务申请的工具是否必要，供主管 Agent 作出授权决定。
   - 在执行后检查结果正确性、完整性、证据和风险。
   - 输出通过或驳回、问题清单、修复要求及返工目标。
   - 图中的评估 Agent与评审 Agent合并为同一能力，不新增工具。

### 控制层

控制层由普通 Python 代码实现，不作为模型可见工具：

- 状态机：约束任务只能按既定阶段流转。
- 调度器：处理依赖、并发上限、超时和取消。
- Agent 注册表：仅供主管 Agent 和控制层使用，记录动态 Agent 的身份、能力、状态和生命周期。
- 工具授权网关：校验调用者、任务编号、工具名称、授权范围和有效期，拒绝一切未授权调用。
- 主管路由器：只允许主管 Agent 与各子 Agent 纵向通信，禁止子 Agent 横向传递消息。
- 重试控制器：限制评审返工次数，防止无限循环。
- 事件记录器：保存任务下发、授权、工具调用、输出、耗时和错误。

### 基础设施层

这些属于内部实现服务，不计入主 Agent 的三个工具：

- 模型客户端。
- 文件读写适配器。
- 子进程运行器。
- 沙箱目录管理器。
- 配置和日志组件。

## 三、完整运行流程

1. 接收任务
   - 创建任务编号。
   - 固化用户目标、约束和验收条件。

2. 主管 Agent 拆解与授权
   - 产生子任务列表。
   - 为每个子任务指定目的、输入、依赖、角色和完成条件。
   - 从三个工具中选择该子任务允许使用的工具子集；默认不授权任何工具。
   - 为每次下发生成不可由子 Agent 修改的授权记录。

3. 首轮评审
   - 主管 Agent 调用评审 Agent检查拆解方案及工具授权是否必要、最小化。
   - 不通过时，根据反馈重新拆解或收紧授权。
   - 连续三次不通过则终止并返回明确原因。

4. 执行任务
   - 主管 Agent 将任务内容、上下文副本和允许工具清单封装后下发。
   - 编码任务可交给代码 Agent。
   - 研究、分析、汇总等任务可交给子 Agent 创建器，由其动态创建专用 Agent。
   - 子 Agent 只能调用本次任务明确授权的工具；未授权工具既不注入其工具集合，也会被授权网关拒绝。
   - 调度器可并发执行互不依赖的子任务，但各子 Agent 相互不可见、不可通信。
   - 每个执行结果只能返回主管 Agent，不能直接发送给其他子 Agent。

5. 汇总结果
   - 主管 Agent 收集各子 Agent 的独立结果。
   - 如需汇总角色，由主管 Agent 通过子 Agent 创建器创建，并只向其提供必要的结果副本和明确授权。
   - 汇总角色不获取其他子 Agent 的身份引用或通信能力。
   - 汇总结果必须保留每项结论对应的来源、执行状态和证据。

6. 最终评审
   - 评审 Agent对照原始目标和验收条件检查结果。
   - 通过则由主管 Agent输出最终结果。
   - 不通过则只返工被指出的任务，不重复执行已经通过的任务。

### 子 Agent失败升级流程

每一个任务拆分版本都按以下固定流程执行：

1. 原始子 Agent执行
   - 单个子 Agent最多执行三次。
   - 三次均未达到完成条件后，可向主管 Agent发送一次 ReplacementRequest。
   - 该申请只能发送一次；重复申请由控制层直接拒绝。

2. 主管 Agent处理替代申请
   - 主管 Agent检查失败证据、已用工具和任务授权。
   - 同意后通过子 Agent创建器生成一个新的替代 Agent，并重新下发同一子任务。
   - 主管 Agent可调整替代 Agent的提示词与授权工具，但不得改变原始主任务。

3. 替代 Agent执行
   - 替代 Agent同样最多执行三次。
   - 替代 Agent的 can_request_replacement 固定为 false，不具备再次申请新 Agent的能力。
   - 替代 Agent仍失败时，只能向主管 Agent返回 TaskDecompositionIssue，说明任务拆分可能存在问题及失败证据。

4. 主管 Agent重新拆分
   - 主管 Agent收到 TaskDecompositionIssue 后重新检查并修改任务拆分。
   - 主管 Agent最多拥有三次拆分修改机会，每次生成新的计划版本并重新经过评审 Agent检查。
   - 每个新计划版本重新创建原始子 Agent；该原始子 Agent仍有一次替代申请资格，但其替代 Agent仍无申请资格。

5. 人工兜底
   - 第三次任务拆分修改后仍无法完成，状态转为 human_required。
   - 控制层生成 HumanEscalationRequest，包含原始任务、所有计划版本、失败记录、工具调用证据和最后评审意见。
   - 首版只定义人工兜底接口、事件和状态，不实现人工工作台、通知或处理流程。

该机制同时限制单 Agent重试、替代链深度和主管重规划次数，避免无限创建 Agent或无限循环。

## 四、核心状态模型

建议任务状态依次为：

- received：已接收。
- planning：正在拆分。
- plan_review：正在评审拆分结果。
- executing：正在执行子任务。
- replacement_requested：原始子 Agent已申请替代执行者。
- replacement_executing：替代 Agent正在执行同一子任务。
- decomposition_issue：替代 Agent失败，等待主管 Agent复查拆分。
- replanning：主管 Agent正在修改任务拆分。
- aggregating：正在汇总结果。
- result_review：正在评审最终结果。
- completed：通过评审并完成。
- human_required：主管 Agent三次修改任务后仍失败，等待人工介入。
- failed：遇到不可恢复的系统错误。
- cancelled：被用户或控制器取消。

每次状态迁移由控制层执行，Agent 只能提出结果或建议，不能自行跳过阶段。

## 五、核心数据结构

### TaskRequest

- task_id
- objective
- constraints
- acceptance_criteria
- original_input

### SubtaskSpec

- subtask_id
- plan_version
- title
- instructions
- role
- dependencies
- allowed_tools
- expected_output
- max_attempts

### ToolGrant

- grant_id
- task_id
- agent_id
- allowed_tools
- issued_by
- issued_at
- expires_at
- revoked

### AgentRecord

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

### ReplacementRequest

- request_id
- task_id
- subtask_id
- requester_agent_id
- failure_summary
- attempt_evidence
- requested_tools
- request_count

### TaskDecompositionIssue

- task_id
- subtask_id
- plan_version
- failed_agent_ids
- failure_reasons
- attempted_approaches
- recommendation

### ExecutionResult

- subtask_id
- success
- output
- evidence
- artifacts
- error
- duration

### ReviewDecision

- stage
- passed
- issues
- required_fixes
- retry_target
- summary

### GeneratedToolManifest

- tool_id
- tool_name
- entry_file
- description
- input_schema
- output_schema
- created_by_task
- review_status
- version

### HumanEscalationRequest

- task_id
- original_request
- plan_history
- agent_failure_history
- tool_call_evidence
- final_review
- created_at

所有 Agent 间通信采用这些结构化对象，避免依赖自由文本猜测状态。

## 六、权限设计

- 主管 Agent拥有三个 Agent 工具的所有权、注册权、分配权和撤销权。
- 子 Agent默认无工具权限；主管 Agent必须在每次任务下发时明确指定允许工具清单。
- 工具权限按任务授予，不自动继承到后续任务，也不能由子 Agent转授。
- 创建子 Agent时只把获批工具注入该 Agent 的工具集合；运行时授权网关再进行一次校验。
- 子 Agent申请额外工具时只能结束当前执行并向主管 Agent返回申请原因，不能自行调用、发现或注册工具。
- 原始子 Agent连续三次失败后只能向主管 Agent申请一次替代 Agent，该权限不可转让或重复使用。
- 替代 Agent在创建时固定移除替代申请能力，不能继续扩展替代链。
- 子 Agent之间没有消息通道、共享会话或彼此引用；替代申请、失败上报和结果交换都必须发送给主管 Agent。
- 代码 Agent只有在主管 Agent授权后，才能使用其内部的项目目录读写能力和沙箱运行能力。
- 代码 Agent生成的新工具只保存为候选代码，不会自动注册、自动执行或自动授予任何子 Agent。
- 候选工具通过评审后仍由主管 Agent决定是否注册，以及在哪个具体任务中授权使用。
- 评审 Agent只有在主管 Agent授权后运行，并保持只读，不修改产物。
- 子 Agent 创建器只能按主管 Agent签发的任务和授权创建 Agent，不能扩大权限、绕过注册表或突破并发上限。
- 每个子任务和每次工具调用必须能追溯到主任务、授权记录和主管 Agent，不允许动态改变主任务。

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

- 使用 asyncio.create_subprocess_exec 创建独立 Python 进程，优先于线程。
- 使用当前虚拟环境的 Python 解释器。
- 将进程工作目录固定到本次运行目录。
- 设置运行超时并在超时后终止进程。
- 捕获标准输出、标准错误、退出码和耗时。
- 每次运行使用新目录，避免不同任务相互覆盖。
- 明确标注该方案只做路径隔离，不限制网络、CPU、内存、系统调用或目录逃逸，因此不能执行不可信代码。

## 八、建议项目结构

```text
so-agent/
  pyproject.toml
  src/
    so_agent/
      main.py
      config.py
      models.py
      context.py
      orchestrator.py
      workflow.py
      tool_packages/
        code_agent/
          __init__.py
          agent.py
          prompt.md
          api_config.yaml
        subagent_creator/
          __init__.py
          agent.py
          prompt.md
          api_config.yaml
        review_agent/
          __init__.py
          agent.py
          prompt.md
          api_config.yaml
        generated_tools/
          __init__.py
          registry.json
          工具名称/
            tool.py
            manifest.yaml
            test_tool.py
      runtime/
        registry.py
        scheduler.py
        sandbox.py
        tool_registry.py
        permissions.py
        supervisor_router.py
        config_loader.py
        events.py
  sandbox/
  tests/
    unit/
    integration/
    end_to_end/
```

工具类包固定包含四个项目包：

1. code_agent
   - agent.py 定义代码 Agent及其内部能力。
   - prompt.md 单独保存代码 Agent提示词。
   - api_config.yaml 保存该 Agent的模型、接口地址、超时和重试配置。

2. subagent_creator
   - agent.py 定义子 Agent创建和调度逻辑。
   - prompt.md 单独保存创建器提示词。
   - api_config.yaml 保存该 Agent独立的模型调用配置。

3. review_agent
   - agent.py 定义拆解评审、授权评审和结果评审逻辑。
   - prompt.md 单独保存评审提示词。
   - api_config.yaml 保存该 Agent独立的模型调用配置。

4. generated_tools
   - 专门保存代码 Agent生成的工具，每个工具使用独立子目录。
   - tool.py 是工具实现，manifest.yaml 描述名称、版本、输入输出和来源任务，test_tool.py 保存最小验证测试。
   - registry.json 只记录候选工具及其评审、注册状态。
   - 该目录是工具代码仓库，不是主管 Agent的第四个工具入口。

配置安全规则：

- API 配置文件只保存 provider、model、base_url、timeout、retry 和密钥环境变量名称。
- API 密钥不写入仓库，由环境变量注入。
- 三个 Agent可使用不同模型或服务地址，但均由统一 config_loader.py 解析和校验。

核心文件职责：

- orchestrator.py：定义主管 Agent，持有且仅持有三个 Agent 工具入口。
- workflow.py：实现状态机、两阶段评审和最多三次返工。
- models.py：保存所有结构化输入输出模型。
- context.py：保存任务上下文、注册表、沙箱路径和事件记录器引用。
- registry.py：由主管 Agent管理动态 Agent的创建、查询、数量限制和释放，不向子 Agent开放查询接口。
- scheduler.py：处理依赖图、隔离并发执行、单 Agent三次尝试、一次替代申请和失败传播。
- sandbox.py：写入代码文件并启动子进程。
- tool_registry.py：保存三个 Agent工具及已获批准的生成工具元数据，所有权归主管 Agent。
- permissions.py：签发、校验和撤销任务级工具授权。
- supervisor_router.py：强制所有 Agent消息、替代申请和任务拆分问题经过主管 Agent，拒绝子 Agent间通信。
- config_loader.py：分别读取三个 Agent包中的 API 配置，不读取或持久化明文密钥。
- 人工兜底仅在 workflow.py 中保留 HumanEscalationHandler 接口和 human_required 状态，不实现外部人工系统。

## 九、SDK 使用边界

- 采用 Python OpenAI Agents SDK。
- 主管 Agent通过 Agent.as_tool 持有三个 Agent工具的唯一注册入口。
- 三个固定 Agent分别从自己的包中加载提示词和 API 配置，避免职责与模型参数相互耦合。
- 动态角色由子 Agent 创建器在运行时构造 Agent，默认 tools 为空，只能注入 ToolGrant 中获批的工具。
- 代码 Agent生成的工具先写入 generated_tools，评审通过且主管 Agent注册后才可进入授权候选集合。
- 不为子 Agent配置 handoff，也不共享 session，避免建立任何横向通信路径。
- 评审 Agent使用 Pydantic 模型输出 ReviewDecision。
- Runner负责单次 Agent 执行，外层 workflow.py 负责授权、路由和业务重试，二者职责分离。
- 每次运行显式设置最大轮次、模型超时和工具超时。
- 在项目独立虚拟环境中安装依赖，避免直接升级当前全局环境中的 openai 包。

## 十、配置建议

- 最大动态 Agent 数量：默认 8。
- 最大并发数：默认 3。
- 子 Agent默认工具数：0。
- 工具授权有效期：仅限单次子任务执行。
- 子 Agent横向通信：始终禁用。
- 单个子 Agent任务执行尝试：最多 3 次。
- 原始子 Agent替代申请：最多 1 次。
- 替代 Agent再次申请权限：禁用。
- 主管 Agent任务拆分修改：最多 3 次。
- 拆解评审重试：包含在三次任务拆分修改额度内。
- 结果评审返工：进入对应子任务的失败升级流程。
- 单个 Agent最大模型轮次：默认 10。
- 模型调用超时：默认 60 秒。
- 沙箱进程超时：默认 30 秒。
- 沙箱输出上限：默认 1 MB，超出截断并记录。

以上参数全部通过配置文件或环境变量覆盖，不写死在提示词中。

## 十一、测试计划

- 单元测试：状态迁移、依赖调度、授权签发与撤销、注册表上限、沙箱超时和各级次数上限。
- 包结构测试：确认工具类包固定包含三个 Agent包和一个生成工具包，三个 Agent包均具有提示词与 API 配置。
- 配置测试：确认三个 API 配置可独立加载、缺失字段能被拒绝且仓库内不存在明文密钥。
- 主体配置测试：确认主管 Agent工具集合严格等于三个指定工具，子 Agent默认工具集合为空。
- 生成工具测试：确认代码 Agent只能写入 generated_tools，候选工具未经评审和主管注册时不可调用。
- 权限测试：确认子 Agent只能调用本任务获批工具，无法转授权限，过期或越权调用必定失败。
- 通信隔离测试：确认子 Agent无法获取彼此引用、共享会话或直接发送消息。
- 替代流程测试：确认原始子 Agent三次失败后只能申请一次，替代 Agent没有申请能力且失败后只能报告任务拆分问题。
- 重规划测试：确认主管 Agent最多修改拆分三次，第四次不会继续运行而是生成 HumanEscalationRequest。
- 人工兜底测试：确认只产生 human_required 状态和完整交接数据，不调用未实现的人工系统。
- 模拟集成测试：使用假模型验证拆解、授权、动态创建、替代、重规划、汇总、驳回和人工兜底流程，不消耗真实 API。
- 沙箱测试：验证代码写入、子进程执行、输出捕获、失败退出和超时终止。
- 真实端到端测试：在提供 API 密钥后执行一个完整任务，并确认最终结果经过两阶段评审。

## 十二、分阶段落地建议

### 最小可运行版本

- 建立项目、虚拟环境和四个工具项目包。
- 为三个固定 Agent分别建立提示词文件和 API 配置文件。
- 定义结构化模型与运行上下文。
- 实现三个固定 Agent工具及生成工具仓库。
- 实现动态 Agent注册表。
- 实现验证型沙箱。
- 实现两阶段评审、子 Agent三次尝试、一次替代申请和主管 Agent三次重规划。
- 定义人工兜底接口与状态，不实现人工处理端。
- 提供命令行入口和模拟测试。

### 稳定性增强

- 增加持久化事件日志、任务恢复和取消机制。
- 增加 token、耗时和并发预算。
- 增加动态 Agent 模板与能力白名单。

### 生产化增强

- 将验证型沙箱替换为 Docker 或远程隔离执行环境。
- 增加可观测性、审计、密钥管理和多租户隔离。
- 增加 Web 控制台和任务可视化。

## 十三、关键假设与风险

- 工作区当前为空，可按全新项目设计，不考虑旧代码兼容。
- 首版使用独立 Python 虚拟环境。
- 代码运行目录并非安全沙箱，只适用于可信测试代码。
- 动态 Agent不会自动成为主管 Agent的新工具，而是在子 Agent创建器内部完成创建和执行。
- generated_tools 中的代码只是候选工具，未经评审和主管 Agent注册不会进入可调用范围。
- 三份 API 配置相互独立，但密钥统一只从环境变量读取。
- 主管 Agent是唯一的工具所有者和消息中枢，子 Agent不能自行扩大权限或相互通信。
- 每个计划版本最多包含一层替代 Agent，不允许替代 Agent继续申请替代者。
- 主管 Agent最多修改任务拆分三次，之后必须进入人工兜底状态而不是继续自动重试。
- 人工兜底在首版中仅为逻辑占位，不包含界面、通知、工单或人工回写能力。
- 模型输出即使使用结构化类型仍可能失败，控制层必须捕获解析错误。
- 模型调用存在成本和不确定性，测试应优先使用假模型。