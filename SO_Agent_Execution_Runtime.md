# SO-Agent Execution Runtime Blueprint (Multi-Agent Framework)

> **Document nature and status-tag convention**: this document preserves the original design rationale from project inception and appends an "Actual evolution" note to each chapter describing the shape after implementation. Capabilities are tagged with four statuses: **[Implemented]** (in current code and test-covered), **[Partially implemented]** (core in place, bounded), **[Deferred]** (design rationale retained, not currently worked on), **[Planned]** (not yet implemented). The Chinese counterpart [多_Agent_框架.md](./多_Agent_框架.md) is section-for-section and capability-status aligned with this document. Test counts and capability claims reflect the current code snapshot (1209 tests) and are not permanent guarantees.

## 0. Implementation Status Overview

**Implemented (1209 tests, current snapshot)**: four-quadrant task routing; thin WorkflowEngine with planning / execution / aggregation coordinators; 13-phase shadow FSM; two-stage review (`passed` as the sole approval field + fail closed + non-blocking `suggestions`); authorization proposals with deferred grant issuance and revoke on all exits; star communication isolation; explicit `reuse_from` reuse; five-layer streaming timeouts; PipelineGate stop gate; tool timeout profiles; SQLite schema sequential incremental migration; FailureClassifier; failure fingerprints and deduplication; RecoveryPolicy; provider no-replacement rule; FORMAT single bounded contract repair then REPLAN; low-trust agent text cannot forge system failures; SQLite persistence (tasks / Execution / PlanVersion / event logs); path-isolated SandboxRunner; MCP protocol layer (protocol / server / client / adapter); dynamic tool generation with test_cases semantic acceptance; persistent scheduling daemon.

**Partially implemented**: validation sandbox (path isolation only, not a security boundary); human escalation (HumanEscalationHandler interface and `human_required` state only, no human-handling endpoint); event recording (structured events + local log files; cross-task durable audit trail not yet implemented).

**Deferred**: multi-dimensional budgets; context governance; production container sandbox (Docker / remote isolation).

**Planned**: unified ModelGateway; execution-level and parameter-level permission hardening; Temporal PoC or migration; durable auditing; OpenTelemetry; dynamic agent templates and capability whitelists; web console and task visualization; tool TTL caching with invocation-count-driven persistence thresholds (the earlier Redis idea); statistics database analysis tables (the earlier MySQL idea).

## I. Design Goals

- [Implemented] The Supervisor Agent owns all tools; its tool catalog is strictly limited to the Code Agent, the Subagent Creator, and the Review Agent.
- [Implemented] The Supervisor Agent is responsible for understanding goals, decomposing tasks, granting tools, dispatching tasks, and organizing the final answer; it does not directly access the file system or launch processes.
- [Implemented] Subagents have no tool capabilities by default; when dispatching a task, the Supervisor Agent specifies the subset of tools allowed for that task, and unauthorized tools are neither visible nor callable.
- [Implemented] Subagents are forbidden from communicating directly, sharing sessions, or invoking one another; all inputs, results, and feedback must pass through the Supervisor Agent.
- [Implemented] Dedicated subagents are created at runtime and released after use, avoiding a large set of predefined fixed agents.
- [Implemented] Review covers task decomposition, authorization scope, and final results, and uses a layered failure-escalation mechanism to bound retries.
- [Implemented (actual attempt count decided by the recovery policy)] Each subagent may attempt execution at most three times; after the original subagent fails, it may request a replacement agent from the Supervisor Agent only once.
- [Implemented] A replacement agent cannot request another replacement agent; if it still fails, a task-decomposition review is triggered.
- [Implemented] The Supervisor Agent may revise the task decomposition at most three times; once the quota is exhausted, the system enters the human-escalation state; the first version only retains the interface and state, without implementing a human-handling system.
- [Partially implemented (path isolation only)] The first-version sandbox is only for validating the toolchain: it writes code into an isolated directory and runs it through a subprocess, without providing true security isolation.

## II. Overall Layering

### Orchestration Layer

The Supervisor Agent sits at the top layer, receives user goals, and produces subtask plans. It exclusively owns the tool catalog and decides whether each subagent may use tools and which tools it may use. The Supervisor Agent itself registers only the following three agent tools:

1. Code Agent [Implemented]
   - Has its own project package, prompt file, and API configuration file.
   - Creates, modifies, and inspects code files.
   - Invokes the internal sandbox runner to verify code when necessary.
   - Writes newly generated tools into the dedicated generated-tools package, never into other agent packages.
   - Returns a change summary, run results, and error information.

2. Subagent Creator [Implemented]
   - Has its own project package, prompt file, and API configuration file.
   - Receives subtask definitions and tool-authorization lists issued by the Supervisor Agent.
   - Dynamically constructs dedicated agents based on role, prompt, inputs, and authorized scope.
   - Subagents have an empty default tool set, mounting only tools explicitly approved by the Supervisor Agent.
   - Schedules dependency-free subtasks for concurrent execution and advances other tasks serially according to dependencies.
   - Exposes no references, sessions, or message channels of other agents to any subagent.
   - Returns each subagent's structured result separately to the Supervisor Agent, which organizes the aggregation.
   - The aggregation agent shown in the diagram is an on-demand role that receives only result copies provided by the Supervisor Agent, does not communicate with other subagents, and is not a fourth fixed tool.

3. Review Agent [Implemented]
   - Has its own project package, prompt file, and API configuration file.
   - Before execution, checks whether the task decomposition is complete, executable, and aligned with the user's goal.
   - Checks whether the tools requested by each subtask are necessary, so that the Supervisor Agent can make authorization decisions.
   - After execution, checks the correctness, completeness, evidence, and risks of the results.
   - Outputs pass or reject, an issue list, required fixes, and rework targets.
   - The evaluation agent shown in the diagram is merged with the Review Agent into a single capability, adding no new tool.

**Actual evolution (orchestration entry)** [Implemented]: a four-quadrant task routing layer (TaskRouter) precedes the full pipeline — a rule-based pre-check (pure math expressions, unambiguous greetings) plus one lightweight LLM classification dispatch tasks along the complexity × duration quadrants into four paths: direct answer (simple + short), deep reasoning (complex + short), persistent task (simple + long, stored and executed by the daemon), and full workflow (complex + long). Classification call failures, parse failures, or invalid results conservatively degrade to the full workflow; `enable_router=False` disables routing entirely. Inside the full workflow there is a fast path for simple tasks: with at most 2 subtasks, no dependencies, all successful, and sufficient outputs, aggregation and result review are skipped (`skip_result_review_for_simple=True`).

### Control Layer

The control layer is implemented in plain Python code and is not exposed as a model-visible tool:

- State machine [Implemented, evolved into a shadow view]: constrains tasks to flow only through the defined phases; the actual implementation is a declarative python-statemachine TaskStateMachine — a 13-phase shadow chart mirroring the authoritative TaskStatus, with all legal transitions declared explicitly, illegal ones rejected and logged (see Chapter IV).
- Scheduler [Implemented]: handles dependencies, concurrency limits, timeouts, and cancellation (runtime/scheduler.py, SubtaskDAGScheduler; hosts policy-driven retries and replacement lifecycle callbacks).
- Agent registry [Implemented]: used only by the Supervisor Agent and the control layer; records the identity, capabilities, status, and lifecycle of dynamic agents.
- Tool authorization gateway [Implemented, evolution in Chapter VI]: validates the caller, task ID, tool name, authorization scope, and validity period, and rejects all unauthorized calls.
- Supervisor router [Implemented]: permits only vertical communication between the Supervisor Agent and each subagent, and forbids horizontal message passing between subagents.
- Retry controller [Implemented, evolved into the deterministic recovery kernel]: bounds rework counts to prevent infinite loops; actually composed of the FailureClassifier (categorization), failure-fingerprint deduplication, and the RecoveryPolicy decision matrix — all logic in the code layer, never modifiable by the LLM (see Chapter III).
- Event recorder [Partially implemented]: stores task dispatch, authorization, tool calls, outputs, durations, and errors; currently structured events + local log files, with the cross-task durable audit trail planned.
- Stop gate (PipelineGate) [Implemented]: after deadline expiry, user cancellation, or human termination, it refuses every new action at the dispatch / retry / replacement / replan / plan-adoption / grant-issuance entries and isolates late results arriving after timeout or cancellation.

### Infrastructure Layer

These are internal implementation services and do not count toward the three tools of the main agent:

- Model client [Implemented]: OpenAI Agents SDK Runner + AsyncOpenAI client; optional per-dimension HTTP timeouts (connect / read / write / pool).
- File read/write adapters [Implemented].
- Subprocess runner [Implemented]: asyncio subprocess execution for generated tools and sandbox commands.
- Sandbox directory manager [Implemented]: path-isolated SandboxRunner.
- Configuration and logging components [Implemented]: global Settings (env prefix `SO_AGENT_`, `.env`, `--config` overrides) + per-agent api_config.yaml + EventLogger.
- Persistence stores [Implemented]: SQLite TaskStore / ResultStore (with the schema versioning and migration mechanism).
- MCP infrastructure [Implemented]: four layers — protocol (contracts and standard error codes), server (MCP Server base and MCPToolServer), client (in-process direct discovery and invocation), adapter (Agent ↔ MCP Server ↔ SDK FunctionTool, generated tool → MCP Server).

**Actual evolution (module mapping)**: the orchestration and control layers are organized as a "thin orchestrator + three coordinators + shared state base" — `workflow.py` (WorkflowEngine: main loop / human escalation / output assembly only), `planning_coordinator.py` (decomposition and plan review), `execution_coordinator.py` (scheduling and failure escalation), `aggregation_coordinator.py` (aggregation and result review), `state_machine.py` (FSM + WorkflowState + domain exceptions + lenient parsing), `workflow_support.py` (WorkflowDeps container, BaseCoordinator, PipelineGate, run_agent_streamed). Coordinators never reference each other directly; they exchange state exclusively through the shared WorkflowState.

## III. Complete Execution Flow

1. Receive task [Implemented]
   - Create a task ID.
   - Fix the user goal, constraints, and acceptance criteria.

2. Supervisor Agent decomposition and authorization [Implemented]
   - Produce a subtask list.
   - Specify the purpose, inputs, dependencies, role, and completion conditions for each subtask.
   - Select the subset of tools allowed for the subtask from the three tools; no tools are authorized by default.
   - Generate an authorization record for each dispatch that subagents cannot modify.

3. First-round review [Implemented]
   - The Supervisor Agent invokes the Review Agent to check the decomposition plan and whether tool authorizations are necessary and minimal.
   - If rejected, re-decompose or tighten authorization based on the feedback.
   - If rejected three consecutive times, terminate and return an explicit reason.

4. Execute tasks [Implemented]
   - The Supervisor Agent packages the task content, context copy, and allowed-tool list before dispatching.
   - Coding tasks may be handed to the Code Agent.
   - Research, analysis, aggregation, and similar tasks may be handed to the Subagent Creator, which dynamically creates dedicated agents.
   - Subagents may call only the tools explicitly authorized for the current task; unauthorized tools are neither injected into their tool sets nor accepted by the authorization gateway.
   - The scheduler may execute mutually independent subtasks concurrently, but subagents are mutually invisible and cannot communicate.
   - Each execution result can only be returned to the Supervisor Agent and cannot be sent directly to other subagents.

5. Aggregate results [Implemented]
   - The Supervisor Agent collects the independent results of each subagent.
   - If an aggregation role is needed, the Supervisor Agent creates it through the Subagent Creator and provides it only with the necessary result copies and explicit authorization.
   - The aggregation role does not obtain identity references or communication capabilities of other subagents.
   - The aggregated result must preserve the source, execution status, and evidence corresponding to each conclusion.

6. Final review [Implemented]
   - The Review Agent checks the results against the original goal and acceptance criteria.
   - If passed, the Supervisor Agent outputs the final result.
   - If rejected, only the flagged tasks are reworked; tasks that already passed are not re-executed.

**Actual evolution**:
- The four-quadrant router precedes the entry (see Chapter II); only the "complex + long" quadrant enters the full flow above.
- Tool authorization is actually issued as deferred grants from proposals [Implemented]: the plan records only authorization intent (proposals); after the plan review passes, the code layer validates and issues ToolGrants uniformly, so unreviewed or rejected intents never produce usable credentials; every exit — completion, replacement, replan adoption, stop gate — revokes outstanding grants (revoke on all exits).
- During re-planning the Supervisor may declare explicit `reuse_from` (an old plan's subtask_id) on new subtasks [Implemented]: admission is decided by the planning coordinator's safety validator; code never reuses automatically by name or fingerprint match; the reuse decision belongs to the Supervisor Agent.

### Subagent Failure Escalation Flow

Each task-decomposition version follows the fixed flow below:

1. Original subagent execution [Implemented]
   - A single subagent is executed at most three times.
   - After three attempts without meeting the completion conditions, it may send one ReplacementRequest to the Supervisor Agent.
   - This request may be sent only once; duplicate requests are rejected directly by the control layer.

2. Supervisor Agent handles the replacement request [Implemented]
   - The Supervisor Agent checks the failure evidence, tools used, and task authorization.
   - If approved, it generates a new replacement agent through the Subagent Creator and re-dispatches the same subtask.
   - The Supervisor Agent may adjust the replacement agent's prompt and authorized tools, but must not change the original main task.
   - [Evolution] In the first version "supervisor approves after checking failure evidence" is auto-approved by the control layer (evidence fully logged with the ReplacementRequest), consuming no extra model calls; provider-type failures (model-vendor side) never trigger replacement.

3. Replacement agent execution [Implemented]
   - The replacement agent is likewise executed at most three times.
   - The replacement agent's can_request_replacement is fixed to false, and it cannot request another new agent.
   - If the replacement agent still fails, it can only return a TaskDecompositionIssue to the Supervisor Agent, explaining that the task decomposition may have problems, together with the failure evidence.

4. Supervisor Agent re-decomposition [Implemented]
   - After receiving the TaskDecompositionIssue, the Supervisor Agent re-examines and revises the task decomposition.
   - The Supervisor Agent has at most three decomposition-revision opportunities; each produces a new plan version that again goes through review by the Review Agent.
   - Each new plan version recreates the original subagent; that original subagent still has one replacement-request entitlement, but its replacement agent still has none.

5. Human escalation [Partially implemented (interface and state)]
   - If the task still cannot be completed after the third task-decomposition revision, the state transitions to human_required.
   - The control layer generates a HumanEscalationRequest containing the original task, all plan versions, failure records, tool-call evidence, and the final review opinion.
   - The first version only defines the human-escalation interface, events, and state; it does not implement a human workstation, notifications, or a handling process.

This mechanism simultaneously bounds single-agent retries, replacement-chain depth, and supervisor re-planning counts, avoiding unbounded agent creation or infinite loops.

**Actual evolution (deterministic failure recovery kernel)** [Implemented]: the attempt caps above are driven by the recovery kernel; all decisions live in the code layer (runtime/failure.py pure functions and data matrices, never modifiable by the LLM):

- FailureClassifier categorizes failures by exception type and error text — network / rate_limit / timeout / permission / budget / provider / format / executor_contract / quality, among others; low-trust sources (agent self-reported text, tool and sandbox output bodies) cannot forge system failure categories — malicious text is classified as quality only; when structured exceptions conflict with executor self-reports, executor metadata wins.
- Every failure yields a stable fingerprint; once the same-fingerprint failure streak reaches the threshold (default 2), retries are suppressed to prevent same-cause spinning; dedup state and retry budgets are maintained in sync.
- RecoveryPolicy maps categories to actions: network / rate_limit back off and retry (base 1 s, cap 30 s, jitter 0.2); format failures allow exactly one contract-repair retry — if repair still fails the flow enters REPLAN (re-planning) and never replacement; provider failures forbid replacement; quality failures use business retries / replacement within the attempt budget.
- System-level consecutive network failures (during supervisor decomposition) escalate to human escalation after the dedicated network-retry budget is exhausted, without consuming replanning quota.

## IV. Core State Model

The recommended task states, in order, are:

- received: accepted.
- planning: decomposing.
- plan_review: reviewing the decomposition result.
- executing: executing subtasks.
- replacement_requested: the original subagent has requested a replacement executor.
- replacement_executing: the replacement agent is executing the same subtask.
- decomposition_issue: the replacement agent failed; awaiting the Supervisor Agent's decomposition review.
- replanning: the Supervisor Agent is revising the task decomposition.
- aggregating: aggregating results.
- result_review: reviewing the final result.
- completed: passed review and completed.
- human_required: still failed after the Supervisor Agent's three task revisions; awaiting human intervention.
- failed: encountered an unrecoverable system error.
- cancelled: cancelled by the user or the controller.

Every state transition is executed by the control layer; agents may only propose results or suggestions and cannot skip phases on their own.

**Actual evolution (shadow FSM)** [Implemented]: TaskStateMachine is a declarative python-statemachine chart tracking the pipeline through 13 high-level phases: idle → classifying → planning → reviewing_plan → granting_permissions → executing → (monitoring / replacing_agent / creating_tool) → aggregating → reviewing_result → (reworking →) completed. It is a shadow view of the authoritative TaskStatus: every real transition dispatches a mapped event; the failed / cancelled / human_required terminal states have no phase of their own, so the shadow chart freezes at the current phase for audit. All legal transitions are declared explicitly; illegal ones are rejected and logged. The replacement chain (replacement_requested → replacement_executing) is mapped in real time by the scheduler via a controlled lifecycle callback — no longer rebuilt after the fact.

## V. Core Data Structures

### TaskRequest [Implemented]

- task_id
- objective
- constraints
- acceptance_criteria
- original_input

### SubtaskSpec [Implemented (with evolved field)]

- subtask_id
- plan_version
- title
- instructions
- role
- dependencies
- allowed_tools
- expected_output
- max_attempts
- reuse_from [Evolution]: explicit reuse declaration whose value is an old plan's subtask_id; filled by the Supervisor only when confident the semantics still apply during re-planning; it is declaration metadata, not part of the semantics, and is excluded from spec_fingerprint; admission is decided by the planning coordinator's safety validator.

### ToolGrant [Implemented]

- grant_id
- task_id
- agent_id
- allowed_tools
- issued_by
- issued_at
- expires_at
- revoked

### AgentRecord [Implemented]

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

### ReplacementRequest [Implemented]

- request_id
- task_id
- subtask_id
- requester_agent_id
- failure_summary
- attempt_evidence
- requested_tools
- request_count

### TaskDecompositionIssue [Implemented]

- task_id
- subtask_id
- plan_version
- failed_agent_ids
- failure_reasons
- attempted_approaches
- recommendation

### ExecutionResult [Implemented]

- subtask_id
- success
- output
- evidence
- artifacts
- error
- duration

### Execution [Evolution, Implemented]

- One record per subtask attempt (distinct from the final ExecutionResult): attempt number, failure category (failure_type, stored as a FailureType.value string), recovery action, fingerprint, duration, and more; persisted with the task in SQLite to support auditing and failure analysis.

### PlanVersion [Evolution, Implemented]

- Plan-version record: version number, the version's SubtaskSpec list, review verdict, and origin (initial / replacement / re-planning); persisted for re-planning comparison and reuse_from validation.

### ReviewDecision [Implemented (with evolved field)]

- stage (intentionally a plain str to leniently accept the LLM's phase values; the caller normalizes to plan_review / result_review)
- passed (the sole approval field; must come with empty required_fixes when true and at least one entry when false — enforced by the caller, failing closed on violation)
- issues
- suggestions [Evolution]: non-blocking improvement suggestions; non-empty suggestions never block approval and are never a rejection reason
- required_fixes
- retry_target
- summary

### ClassificationResult [Evolution, Implemented]

- task_type (direct_answer / deep_reasoning / persistent_task / full_workflow)
- reasoning (one-sentence rationale; carries the degradation cause on classification failure)

### GeneratedToolManifest [Implemented (with evolved fields)]

- tool_id
- tool_name
- entry_file
- description
- input_schema
- output_schema
- created_by_task
- review_status
- version
- test_cases / expected_outputs [Evolution]: acceptance test-case arrays paired with expected return values; before registration the system runs each case in the sandbox and compares outputs semantically — only fully matching tools pass acceptance and register.

### HumanEscalationRequest [Implemented]

- task_id
- original_request
- plan_history
- agent_failure_history
- tool_call_evidence
- final_review
- created_at

All inter-agent communication uses these structured objects, avoiding reliance on free text to guess state.

## VI. Permission Design

- [Implemented] The Supervisor Agent holds the ownership, registration, assignment, and revocation rights for the three agent tools.
- [Implemented] Subagents have no tool permissions by default; the Supervisor Agent must explicitly specify the allowed-tool list on every task dispatch.
- [Implemented] Tool permissions are granted per task; they are not automatically inherited by subsequent tasks and cannot be re-delegated by subagents.
- [Implemented] When creating a subagent, only the approved tools are injected into that agent's tool set; the authorization gateway performs one more validation at runtime.
- [Implemented] When a subagent requests additional tools, it can only end the current execution and return the request rationale to the Supervisor Agent; it cannot call, discover, or register tools on its own.
- [Implemented] After three consecutive failures, the original subagent may request a replacement agent from the Supervisor Agent only once; this entitlement is non-transferable and non-reusable.
- [Implemented] A replacement agent has its replacement-request capability removed at creation time and cannot extend the replacement chain further.
- [Implemented] Subagents have no message channels, shared sessions, or mutual references; replacement requests, failure reports, and result exchanges must all be sent to the Supervisor Agent.
- [Implemented] The Code Agent may use its internal project-directory read/write capability and sandbox execution capability only after authorization by the Supervisor Agent.
- [Implemented] New tools generated by the Code Agent are stored only as candidate code; they are never automatically registered, automatically executed, or automatically granted to any subagent.
- [Implemented] Even after a candidate tool passes review, the Supervisor Agent still decides whether to register it and in which specific task to authorize its use.
- [Implemented] The Review Agent runs only after authorization by the Supervisor Agent and remains read-only, without modifying artifacts.
- [Implemented] The Subagent Creator may create agents only according to the tasks and authorizations issued by the Supervisor Agent; it cannot expand permissions, bypass the registry, or exceed concurrency limits.
- [Implemented] Every subtask and every tool call must be traceable to the main task, the authorization record, and the Supervisor Agent; the main task may not be changed dynamically.

**Actual evolution (grant lifecycle)** [Implemented]:

- Deferred issuance from proposals: the plan records authorization intent (proposals); after the plan review passes, the permission gateway validates and issues ToolGrants uniformly (with expires_at); unreviewed or rejected intents never yield usable credentials.
- Revoke on all exits: task completion / failure / cancellation, replacement-chain triggers, adoption of a new plan version, and PipelineGate stops all revoke outstanding grants (revoke_grants_for_agent / revoke_grants_for_task), with revocation reasons logged.
- Current grant granularity is tool-name level; execution-level (call-context) and parameter-level (input-constraint) hardening remain planned.

## VII. Validation Sandbox

Recommended directory layout:

```text
sandbox/
  task_<task_id>/
    run_<run_id>/
      main.py
      stdout.txt
      stderr.txt
      metadata.json
```

Execution mechanism:

- Use asyncio.create_subprocess_exec to create an independent Python process, preferred over threads. [Implemented]
- Use the Python interpreter of the current virtual environment. [Implemented]
- Pin the process working directory to the current run directory. [Implemented]
- Set a run timeout and terminate the process after it expires. [Implemented]
- Capture standard output, standard error, exit code, and duration. [Implemented]
- Use a fresh directory for each run to prevent different tasks from overwriting each other. [Implemented]
- Explicitly mark that this approach provides only path isolation and does not restrict network, CPU, memory, system calls, or directory escape; therefore, it must not execute untrusted code. [Implemented (this limit is a factual constraint; see the risk statement)]

**Actual evolution and deferral**: the earlier "sandbox network whitelist" idea (allowing some domains for generated tools' networking needs) was evaluated and rejected, since isolation cannot be guaranteed once any network exception exists; it has been merged into the deferred "production container sandbox" — container-level isolation (Docker / remote execution) retains its design rationale as a deferred item, and the current SandboxRunner is suitable only for trusted test code.

## VIII. Project Structure (Actual)

```text
so-agent/
  pyproject.toml
  secrets.yaml              # key injection source (not committed; loaded into env by main.py)
  src/
    so_agent/
      main.py               # CLI entry (one-shot tasks / --daemon scheduler)
      config.py             # global Settings (SO_AGENT_ env prefix / .env / --config)
      context.py            # ProjectContext (sandbox dir, project name, config)
      orchestrator.py       # Supervisor Agent definition and OrchestratorOutput
      workflow.py           # WorkflowEngine thin orchestrator (main loop / escalation / facade proxies)
      workflow_support.py   # WorkflowDeps / BaseCoordinator / PipelineGate / run_agent_streamed
      state_machine.py      # TaskStateMachine (13-phase shadow FSM) / WorkflowState / domain exceptions
      planning_coordinator.py
      execution_coordinator.py
      aggregation_coordinator.py
      router.py             # TaskRouter four-quadrant routing
      paths/                # direct_answer / deep_reasoning / persistent_task
      models.py             # all structured input/output models
      fingerprint.py        # spec fingerprints
      json_utils.py         # lenient JSON extraction
      persistence.py        # SQLite TaskStore
      result_store.py       # ResultStore (in-memory / SQLite) and result-fingerprint storage
      schema_migrations.py  # schema versioning and sequential incremental migration
      task_scheduler.py     # PeriodicTaskPoller daemon
      api_config.yaml(.example)
      mcp/                  # protocol / server / client / adapter
      runtime/
        config_loader.py    # api_config.yaml loading and per-dimension HTTP timeout resolution
        events.py           # EventLogger structured events
        failure.py          # FailureClassifier / fingerprints / RecoveryPolicy
        permissions.py      # PermissionGateway (propose / grant / revoke)
        registry.py         # AgentRegistry
        sandbox.py          # SandboxRunner (path isolation)
        scheduler.py        # SubtaskDAGScheduler (policy-driven retries)
        supervisor_router.py # star-communication enforcement
        tool_registry.py    # tool catalog (three agent tools + registered generated tools)
        generated_tool_runner.py # generated-tool subprocess execution and semantic acceptance
      tool_packages/
        code_agent/         # agent.py + prompt.md + api_config.yaml
        review_agent/       # agent.py + prompt.md + api_config.yaml
        subagent_creator/   # agent.py + prompt.md + api_config.yaml
        generated_tools/    # generated-tool repository (one subdirectory per tool + registry.json)
  sandbox/                  # runtime workspace (tasks / sandbox run dirs / task store)
  tests/
    unit/  integration/  end_to_end/  baseline/
```

Configuration security rules [Implemented]:

- API configuration files store provider, model, base_url, timeout, retry (plus optional per-dimension HTTP timeouts) and the key environment variable name (api_key_env).
- API keys are never committed: injected via the api_key_env environment variable or the root secrets.yaml; writing the key directly as api_key is backward compatibility only and is discouraged.
- The three agents may use different models or endpoints, but all are parsed and validated by the unified config_loader.py; the api_key field is excluded from serialization and never logged.

Core file responsibilities:

- orchestrator.py: defines the Supervisor Agent, which holds exactly and only the three agent tool entries.
- workflow.py: thin orchestrator — main loop, state dispatch, human escalation, and output assembly; two-stage review and failure escalation belong to the coordinators.
- planning_coordinator.py / execution_coordinator.py / aggregation_coordinator.py: the planning / execution / aggregation coordinators.
- state_machine.py: 13-phase shadow FSM, shared mutable WorkflowState, workflow domain exceptions, and lenient structured-output parsing.
- workflow_support.py: coordinator dependency container, PipelineGate stop gate, and the five-layer streaming executor run_agent_streamed.
- router.py: four-quadrant classification and dispatch.
- persistence.py / result_store.py / schema_migrations.py: SQLite persistence and version migration.
- runtime/failure.py: failure classification, fingerprints, and recovery policy (code-only decisions).
- runtime/permissions.py: authorization proposals, issuance, validation, and revocation.
- runtime/scheduler.py: dependency-graph scheduling, isolated concurrency, policy-driven retries, and replacement lifecycle.
- runtime/sandbox.py: path-isolated subprocess execution.
- runtime/config_loader.py: api_config.yaml loading, key resolution, and HTTP timeout resolution at every level.
- Human escalation retains only the HumanEscalationHandler interface and the human_required state in workflow.py / workflow_support.py, without implementing an external human system.

## IX. SDK Usage Boundaries

- Use the Python OpenAI Agents SDK. [Implemented]
- The Supervisor Agent holds the sole registration entry for the three agent tools through Agent.as_tool. [Implemented]
- The three fixed agents each load prompts and API configuration from their own package, avoiding coupling between responsibilities and model parameters. [Implemented]
- Dynamic roles are constructed at runtime by the Subagent Creator with an empty default tools list; only tools approved in a ToolGrant can be injected. [Implemented]
- Tools generated by the Code Agent are first written into generated_tools and enter the authorization candidate set only after passing review and being registered by the Supervisor Agent. [Implemented]
- Do not configure handoffs for subagents or share sessions, avoiding any horizontal communication path. [Implemented]
- The Review Agent uses Pydantic models to output ReviewDecision. [Implemented]
- The Runner is responsible for a single agent execution, while the outer workflow handles authorization, routing, and business retries; the two responsibilities are separated. [Implemented]
- Explicitly set the maximum turns, model timeout, and tool timeout for each run. [Implemented]
- Install dependencies in the project's dedicated virtual environment; avoid directly upgrading the openai package in the current global environment. [Implemented]

**Actual evolution (streaming execution and timeouts)** [Implemented]:

- Agent calls uniformly run through run_agent_streamed with five layers: first_event_timeout (first event) → model_idle_timeout (model idle, renewed only by native SDK events) → tool_execution_timeout (standard tool window) → agent_total_timeout / supervisor_total_timeout (per-call totals) → max_pipeline_timeout (pipeline deadline with graceful degradation on expiry).
- Nested subagents and the long-running code agent use dedicated tool windows (derived by default as agent_total_timeout + 10 s cleanup grace); totals are still truncated at runtime by the remaining pipeline budget.
- HTTP client timeouts and streaming-phase timeouts are different things: per-dimension connect / read / write / pool timeouts only make individual I/O operations fail fast; total duration is bounded by the upper-layer windows.
- The CLI disables SDK trace export by default (OPENAI_AGENTS_DISABLE_TRACING=1) to remove warnings when no Tracing key is present.

## X. Configuration (Actual Defaults)

The values below are the real defaults in the current code (config.py), all overridable via environment variables (prefix `SO_AGENT_`), `.env`, or a `--config` file, never hard-coded in prompts. They are runtime time caps and governance quotas and **do not constitute an implemented multi-dimensional budget system** (budget and context governance are deferred).

| Parameter | Default | Description |
| --- | --- | --- |
| max_dynamic_agents | 8 | Max dynamically created subagents |
| max_agent_attempts | 3 | Max attempts per agent (actual count decided by the recovery policy) |
| max_replacement_requests | 1 | Max replacement requests (replacement-chain depth 1) |
| max_replan_attempts | 3 | Max decomposition revisions by the Supervisor |
| max_concurrency | 3 | Max concurrent subtasks |
| max_model_turns | 8 | Max model turns per agent run |
| model_timeout | 45 s | Per model-call timeout (HTTP layer) |
| first_event_timeout | 180 s | Streaming first-event window |
| model_idle_timeout | 90 s | Model-phase idle window |
| tool_execution_timeout | 180 s | Standard tool execution window |
| subagent_tool_execution_timeout | 610 s | Nested-subagent tool window (derived = agent_total + 10) |
| code_agent_tool_execution_timeout | 610 s | Code-agent tool window (derived = agent_total + 10) |
| agent_total_timeout | 600 s | Total budget per agent call |
| supervisor_total_timeout | 900 s | Total budget per supervisor call (floored at the nested tool window) |
| max_pipeline_timeout | 2400 s | Pipeline deadline (graceful degradation on expiry) |
| sandbox_timeout | 30 s | Sandbox per-command timeout |
| sandbox_output_limit | 1 MiB | Sandbox output cap |
| classifier_max_tokens | 512 | Max output tokens for the routing classifier call |
| enable_router | true | Four-quadrant router switch (False runs the full workflow) |
| skip_result_review_for_simple | true | Simple-task fast path (skips aggregation and result review) |
| recovery_backoff_base_seconds | 1.0 | network/rate_limit backoff base |
| recovery_backoff_cap_seconds | 30.0 | Single backoff cap |
| recovery_backoff_jitter_ratio | 0.2 | Backoff jitter ratio |
| recovery_retry_suppress_threshold | 2 | Same-fingerprint consecutive-failure suppression threshold |
| recovery_repair_contract_max_attempts | 1 | FORMAT contract-repair retry cap |
| task_store_path | sandbox/task_store.db | SQLite task store path |
| scheduler_poll_interval | 60 s | Daemon polling interval |

Subagents default to 0 tools; tool authorization is valid for a single subtask execution only; subagent horizontal communication is always disabled. The legacy timeout fields (max_idle_timeout / max_supervisor_call_timeout) are compatibility aliases only, mapped to the new fields when explicitly set.

## XI. Test Plan

- [Implemented] Unit tests: state transitions, dependency scheduling, authorization issuance and revocation, registry limits, sandbox timeouts, and count limits at each level.
- [Implemented] Package-structure tests: confirm that the tool-class package fixedly contains three agent packages and one generated-tools package, and that all three agent packages have prompts and API configuration.
- [Implemented] Configuration tests: confirm that the API configurations at every level load independently, that missing fields are rejected, and that no plaintext keys exist in the repository.
- [Implemented] Main-configuration tests: confirm that the Supervisor Agent's tool set strictly equals the three designated tools and that the default tool set of subagents is empty.
- [Implemented] Generated-tool tests: confirm that the Code Agent can write only into generated_tools and that candidate tools are not callable before review and supervisor registration; semantic acceptance compares per case.
- [Implemented] Permission tests: confirm that subagents can call only tools approved for the current task, cannot re-delegate permissions, and that expired or over-privileged calls always fail; deferred proposal issuance and revoke-on-all-exits have dedicated coverage.
- [Implemented] Communication-isolation tests: confirm that subagents cannot obtain references to one another, share sessions, or send messages directly.
- [Implemented] Replacement-flow tests: confirm that the original subagent can request a replacement only once after three failures, that the replacement agent has no request capability, and that after failing it can only report a task-decomposition issue; the provider no-replacement rule has dedicated coverage.
- [Implemented] Re-planning tests: confirm that the Supervisor Agent may revise the decomposition at most three times, and that on the fourth time it does not continue running but instead generates a HumanEscalationRequest; explicit reuse_from validation has dedicated coverage.
- [Implemented] Human-escalation tests: confirm that only the human_required state and complete handover data are produced, without invoking an unimplemented human system.
- [Implemented] Failure-recovery tests: classification matrix, fingerprint deduplication, injected backoff, FORMAT repair then REPLAN, and low-trust text cannot forge system failures.
- [Implemented] Timeout and stop-gate tests: five-layer streaming timeouts, stop-gate refusal of new actions, and late-result isolation.
- [Implemented] Schema migration tests: sequential incremental migration, idempotency, and fail closed on unknown higher versions.
- [Implemented] Simulated integration tests: use fake models to verify decomposition, authorization, dynamic creation, replacement, re-planning, aggregation, rejection, and human-escalation flows, without consuming the real API.
- [Implemented] Sandbox tests: verify code writing, subprocess execution, output capture, failure exit, and timeout termination.
- [Partially implemented] Real end-to-end tests: after an API key is provided, execute one complete task and confirm that the final result has gone through two-stage review; these require a real key and are excluded from the default suite.

**Current status**: the suite currently collects **1209 tests (current snapshot)**, all passing offline; the number changes as the code evolves and is not a permanent capability guarantee.

## XII. Phased Rollout (Progress Mapping)

### Minimal Runnable Version [Implemented]

- Set up the project, the virtual environment, and the four tool project packages. ✅
- Create prompt files and API configuration files for each of the three fixed agents. ✅
- Define the structured models and the runtime context. ✅
- Implement the three fixed agent tools and the generated-tools repository. ✅
- Implement the dynamic agent registry. ✅
- Implement the validation sandbox. ✅ (path isolation)
- Implement two-stage review, three subagent attempts, one replacement request, and three supervisor re-plannings. ✅
- Define the human-escalation interface and state, without implementing the human-handling endpoint. ✅
- Provide a command-line entry point and simulated tests. ✅

### Stability Enhancements [Partially implemented]

- Persistent event logs, task recovery, and cancellation mechanisms [Implemented]: EventLogger structured events, the SQLite task store with daemon, PipelineGate, and cooperative cancellation. The cross-task durable audit trail remains planned.
- Token, duration, and concurrency budgets [Partially implemented / partially deferred]: concurrency limits and the full set of time caps are implemented with basic token usage recorded; the multi-dimensional budget system is deferred.
- Dynamic agent templates and a capability whitelist [Planned].

### Production Enhancements [Planned / partially deferred]

- Replace the validation sandbox with Docker or a remote isolated execution environment [Deferred (design retained)].
- Observability (OpenTelemetry), durable auditing, and multi-tenant isolation [Planned]; key management [Implemented (centralized secrets.yaml injection + environment variables)].
- A web console and task visualization [Planned].

## XIII. Key Assumptions and Risks

- [Archived: inception-time assumption, no longer applicable] The workspace is currently empty and can be designed as a greenfield project, without considering legacy-code compatibility.
- [Implemented] The project uses a dedicated Python virtual environment.
- [Risk, must-know] The code execution directory is not a secure sandbox and is only suitable for trusted test code; it cannot safely run untrusted code.
- [Implemented] Dynamic agents do not automatically become new tools of the Supervisor Agent; creation and execution are completed inside the Subagent Creator.
- [Implemented] Code in generated_tools is merely candidate tools and will not enter the callable scope without review and registration by the Supervisor Agent.
- [Implemented] The API configurations at every level are mutually independent, but keys are uniformly read only from environment variables (or injected via secrets.yaml).
- [Implemented] The Supervisor Agent is the sole tool owner and message hub; subagents cannot expand their own permissions or communicate with each other.
- [Implemented] Each plan version contains at most one layer of replacement agent; a replacement agent is not allowed to request further replacements.
- [Implemented] The Supervisor Agent may revise the task decomposition at most three times, after which it must enter the human-escalation state rather than continue automatic retries.
- [Partially implemented] Human escalation is only a logical placeholder, with no UI, notifications, tickets, or human write-back capability.
- [Implemented] Model outputs may still fail even with structured types; the control layer must catch parsing errors (lenient parsing + fail closed).
- [Implemented] Model calls incur cost and uncertainty; tests should prefer fake models.
- [Risk] This project is experimental / research software; its governance and recovery mechanisms are untested under production load.
- [Risk] Streaming cancellation may be delayed underneath: timeout / cancellation teardown depends on the underlying SDK's cancellation behavior with a bounded close grace; late results are isolated by the stop gate, but prompt resource release is not guaranteed.
- [Deferred] While multi-dimensional budgets and context governance are deferred, cost and context control for long tasks relies on time caps and bounded retry quotas.
