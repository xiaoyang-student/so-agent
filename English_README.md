<div align="right">
  <a href="README.md">
    <img src="https://img.shields.io/badge/lang-中文-red.svg" alt="中文">
  </a>
</div>

<h1 align="center">SO-Agent</h1>
<p align="center">
  <b>A Deterministic, Zero-Trust Multi-Agent Runtime</b><br>
  <i>LLMs think, Code decides.</i>
</p>

# SO-Agent

A governed multi-agent execution runtime: a two-tier, star-topology multi-agent orchestration framework built on the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python). The core design principle is **LLM-as-Thinker, Code-as-Decider** — the LLM understands goals, decomposes tasks, generates instructions, and analyzes content; all decisions such as state transitions, permission issuance, retry control, and execution scheduling are made by the deterministic code layer.

> **Project status: experimental / research use.** Not production-ready; interfaces and behavior may change at any time. Test counts and capability claims reflect the current code snapshot and are not permanent guarantees.

- Architecture document (English): [SO_Agent_Execution_Runtime.md](./SO_Agent_Execution_Runtime.md)
- 中文架构文档：[多_Agent_框架.md](./多_Agent_框架.md)

# Architecture Overview

The diagram below provides an overview of the current implementation architecture, covering implemented capabilities only. Requests enter through a four-quadrant routing layer. The full workflow is driven by a thin orchestrator and three coordinators for planning, execution, and aggregation.

The implemented capabilities include two-stage reviews; the complete lifecycle from authorization proposals to ToolGrant issuance; deterministic failure recovery; star-topology isolation for sub-agents; dynamic tool creation and sandbox acceptance testing; and a cross-cutting control plane spanning the entire workflow, including a stop gate, a shadow state machine, audit events, ResultStore with schema migrations, and human escalation.

Solid lines represent the primary task flow. Dashed lines represent conditional paths and intervention points for the cross-cutting control plane. Deferred and planned directions—including multidimensional budgets, context governance, production container sandboxes, Temporal, a unified ModelGateway, and OpenTelemetry—are excluded from the diagram. See Capability Status and Roadmap below for details.

flowchart TB
    subgraph ENTRY["Entry Points and Four-Quadrant Routing"]
      CLI["CLI: One-Shot Tasks / --daemon Mode"]
      ROUTER["TaskRouter: Rule-Based Preclassification + One Lightweight LLM Classification Call"]
      Q1["Direct Response"]
      Q2["Deep Reasoning"]
      Q3["Persistent Tasks: SQLite + Daemon Scheduling"]
      Q4["Full Workflow"]
    end
    ENGINE["WorkflowEngine: Thin Orchestrator (Main Loop / Output Assembly)"]
    subgraph PLANNING["Planning: PlanningCoordinator"]
      SUP["Supervisor Agent: Task Decomposition / Replanning (LLM)"]
      PROP["Authorization Proposal (Candidate Only; Not Yet Effective)"]
      REV1["Review Agent: Plan Review + Authorization Review (LLM)"]
      CONTRACT1["passed Contract Validation (Fail-Closed)"]
      ISSUE["PermissionGateway: Issue ToolGrant"]
      REUSE["Explicit Result Reuse: reuse_from + Code Safety Validation"]
      REPLAN["Bounded Replanning (≤3 Attempts)"]
    end
    subgraph EXECUTION["Execution: ExecutionCoordinator + Scheduler"]
      SCHED["Execution Scheduling: Dependencies / Concurrency Limits / Five-Layer Timeouts / Cancellation"]
      STAR["Star-Topology Isolated Sub-Agents (LLM): Supervisor as the Sole Intermediary"]
      MCPHUB["Four-Layer MCP + ToolRegistry"]
      GAP["Tool Gap: Dynamic Creation → Sandbox Acceptance Testing → Registration → Supplemental Authorization"]
      CLASSIFY["FailureClassifier: Classification + Fingerprint-Based Deduplication"]
      POLICY["RecoveryPolicy: Decision Matrix (Provider Substitution Prohibited)"]
      FIX["FORMAT: One-Time Contract Repair"]
      REPLACE["Replacement: Maximum Depth 1, No Further Replacement Requests"]
    end
    subgraph AGGREGATION["Aggregation: AggregationCoordinator"]
      SUM["Result Aggregation (LLM-Generated)"]
      REV2["Review Agent: Result Review (LLM)"]
      CONTRACT2["passed Contract Validation (Fail-Closed)"]
      REWORK["Rework: Rerun Only the Subtasks Identified in Review Feedback"]
    end
    subgraph CONTROL["Cross-Cutting Control Plane (Code-Enforced Throughout)"]
      GATE["PipelineGate: Deadline / Cancellation → Reject New Actions and Isolate Late Results"]
      FSM["TaskStateMachine: 13-Stage Shadow FSM"]
      AUDIT["EventLogger: Audit Events + Local Logs"]
      STORE["ResultStore: Execution Records + Plan Versions"]
      MIG["SQLite Schema: Ordered Incremental Migrations, Fail-Closed"]
    end
    subgraph FINALS["Terminal States"]
      DONE["completed"]
      HUMAN["human_required: HumanEscalationRequest Handoff"]
      STOPPED["failed / cancelled"]
    end
    CLI --> ROUTER
    ROUTER -->|"Simple + Short"| Q1
    ROUTER -->|"Complex + Short"| Q2
    ROUTER -->|"Simple + Long-Running"| Q3
    ROUTER -->|"Complex + Long-Running; Fallback on Classification Failure"| Q4
    Q4 --> ENGINE
    ENGINE --> SUP
    SUP --> PROP
    PROP --> REV1
    REV1 --> CONTRACT1
    CONTRACT1 -->|"Both Checks Passed"| ISSUE
    CONTRACT1 -->|"Validation Failed: Consume Decomposition Budget"| SUP
    ISSUE -->|"Authorization Ready"| SCHED
    ISSUE -.->|"Adopt Plan: Validate reuse_from"| REUSE
    REUSE -.->|"Validation Passed: Reuse Previous Results"| SCHED
    REPLAN -->|"Decompose Again"| SUP
    REPLAN -.->|"Budget Exhausted"| HUMAN
    SCHED --> STAR
    STAR -->|"Invoke Authorized Tools via MCP"| MCPHUB
    SCHED -.->|"Plan References an Unregistered Tool"| GAP
    GAP -->|"Register as a New Tool"| MCPHUB
    GAP -->|"Retry After Supplemental Authorization"| SCHED
    STAR -->|"Failure"| CLASSIFY
    CLASSIFY --> POLICY
    POLICY -->|"network / rate_limit: Retry with Backoff"| SCHED
    POLICY -->|"quality / tool / sandbox: Replace After Retries Are Exhausted"| REPLACE
    POLICY -->|"format"| FIX
    FIX -->|"Repair Still Fails"| REPLAN
    REPLACE -->|"Replacement Still Fails: Decompose the Problem"| REPLAN
    STAR -->|"Subtask Completed"| SUM
    SUM --> REV2
    REV2 --> CONTRACT2
    CONTRACT2 -->|"passed"| DONE
    CONTRACT2 -->|"Validation Failed"| REWORK
    REWORK -->|"Rerun Identified Subtasks"| SCHED
    GATE -.->|"Stop-Gate Checks"| ENGINE
    FSM -.->|"State Transition Mapping"| ENGINE
    AUDIT -.->|"End-to-End Audit Trail"| ENGINE
    STORE -.->|"Persist Results / Versions"| SCHED
    MIG -.->|"Versioned Migrations"| STORE
    ENGINE -.->|"Unrecoverable Error / Cancellation Cleanup"| STOPPED

Responsibility Boundaries

LLMs—including the supervisor, review agents, and sub-agents—are responsible only for semantic understanding, task decomposition during planning, review feedback, and content generation.

All task state transitions, permission issuance and revocation, scheduling, timeout enforcement, cancellation, failure recovery policies, and the final passed release decision are determined by the deterministic code layer.

Review decisions take effect only after passing contract validation in the code layer. The validation process follows a fail-closed policy: a review decision that fails validation cannot authorize the workflow to proceed.

Capability Status and Roadmap

The diagram intentionally includes only implemented capabilities. The following directions are deferred or planned and are not represented as implemented functionality:

Multidimensional budgets
Context governance
Production-grade container sandboxing
Temporal-based workflow orchestration
A unified ModelGateway
OpenTelemetry integration

## Core Features (Implemented)

All capabilities below are implemented in the current code and covered by **1209 tests (current snapshot)**:

- **Four-quadrant task routing**: rule-based pre-check plus one lightweight LLM classification; tasks are dispatched along the complexity × duration quadrants to direct answer / deep reasoning / persistent task / full workflow paths. Classification failure conservatively degrades to the full workflow.
- **Two-tier star architecture**: the Supervisor Agent exclusively holds exactly three tool entries (code_agent, subagent_creator, review_agent); subagents have no message channels or shared sessions — all communication is relayed by the Supervisor.
- **Thin orchestrator + three coordinators**: WorkflowEngine keeps only the main loop and human escalation; planning (decomposition and plan review), execution (scheduling and failure escalation), and aggregation (synthesis and result review) belong to three coordinators exchanging state via a shared state object.
- **13-phase shadow FSM**: a python-statemachine declarative state chart mirrors the authoritative task status; all legal transitions are declared explicitly, illegal ones are rejected and logged.
- **Two-stage review**: plan review and result review both produce a structured ReviewDecision; `passed` is the only approval field and the gate is fail closed — an invalid contract or parse failure is conservatively treated as rejection; `suggestions` are non-blocking and never a rejection reason.
- **Authorization proposal with deferred issuance and revoke-on-all-exits**: tool grants are issued by the code layer only after the plan review passes (proposal → grant); every exit — task completion, replacement, re-planning — revokes outstanding grants.
- **Deterministic failure recovery**: FailureClassifier categorizes failures (low-trust agent text cannot forge system failure categories), failure fingerprints deduplicate retries (retries suppressed once the same-cause streak reaches the threshold), and RecoveryPolicy decides retry / repair / replacement / replanning per category; provider-type failures forbid replacement, and FORMAT failures allow exactly one contract repair before entering REPLAN.
- **Replacement-chain depth 1 and bounded replanning**: up to 3 attempts per agent, the original subagent may request a replacement only once, replacement agents cannot request further replacements, and the Supervisor may revise the decomposition at most 3 times (with explicit `reuse_from` reuse of prior results) before entering the `human_required` state.
- **Five-layer streaming timeouts and stop gate**: first-event / model-idle / tool-execution / per-agent total / pipeline deadline windows; the PipelineGate rejects every new action after deadline or cancellation and isolates late results.
- **Tool timeout profiles**: standard tools, nested subagents, and the long-running code agent use dedicated tool-phase wait windows so long tasks are not killed prematurely by the generic window.
- **Dynamic tool generation and MCP layer**: tools generated by code_agent must prove correctness via test_cases and pass per-case semantic verification in the sandbox before registration; the protocol layer provides an MCP server, client, and bidirectional adapters.
- **Persistence and schema migration**: SQLite stores long-running tasks, execution records (Execution), and plan versions (PlanVersion); versioned sequential incremental migrations are idempotent, transactional, and fail closed on unknown higher schema versions.

## Installation

Requires Python 3.10+. All commands below are valid in Windows PowerShell (no `&&` separators).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Key Configuration

Keys are never committed to the repository. Two options (either one):

1. **Environment variables**: set the variable referenced by each agent's `api_key_env`, for example:

   ```powershell
   $env:SO_AGENT_API_KEY = "sk-your-key"
   ```

2. **secrets.yaml** (project root, ignored by version control): `main.py` loads its key-value pairs into environment variables at startup.

Connection settings live in `src/so_agent/api_config.yaml` and `src/so_agent/tool_packages/<agent>/api_config.yaml`. The recommended style is to declare `api_key_env` (the environment variable name); writing the key directly as `api_key` is kept only for backward compatibility and is discouraged. See the `api_config.yaml.example` files at each level.

## Quick Start

```powershell
# One-shot task (the objective may also be piped in via stdin)
.\.venv\Scripts\python.exe -m so_agent.main "Implement a CLI statistics tool for the target directory" --constraints "Standard library only" --acceptance-criteria "Provide a runnable self-verifying script"

# Persistent task scheduler daemon (polls and executes due tasks in the background; no objective needed)
.\.venv\Scripts\python.exe -m so_agent.main --daemon
```

Notes:

- `--config <file>` overrides `Settings` defaults with a YAML / JSON file (e.g. `max_concurrency`, `model_timeout`);
- Exit code is 0 when the task reaches the `completed` terminal state, 1 for other terminal states (`human_required` / `failed` / `cancelled`), and 2 for argument or configuration errors;
- Standard output prints only the structured JSON result; run-status summaries go to standard error.

## Running the Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The suite currently collects **1209 tests (current snapshot)**, all using fake models with no real API consumption; real end-to-end tests require a key and are executed separately.

## Key Default Settings (Verified Against Current Code)

The values below are runtime time caps and governance parameters, overridable via environment variables (`SO_AGENT_` prefix), `.env`, or a `--config` file. They are **not** an implemented multi-dimensional budget system (budget and context governance are deferred — see below).

| Parameter | Default | Description |
| --- | --- | --- |
| `max_model_turns` | 8 | Max model turns per agent run |
| `model_timeout` | 45 s | Per model-call timeout (HTTP layer) |
| `first_event_timeout` | 180 s | Streaming first-event window |
| `model_idle_timeout` | 90 s | Model-phase idle window |
| `tool_execution_timeout` | 180 s | Standard tool execution window |
| `subagent_tool_execution_timeout` | 610 s | Nested-subagent tool window (derived by default = agent total + 10 s) |
| `code_agent_tool_execution_timeout` | 610 s | Code-agent tool window (same derivation) |
| `agent_total_timeout` | 600 s | Total budget per agent call |
| `supervisor_total_timeout` | 900 s | Total budget per supervisor call |
| `max_pipeline_timeout` | 2400 s | Pipeline deadline (graceful degradation on expiry) |
| `max_dynamic_agents` | 8 | Max dynamically created subagents |
| `max_concurrency` | 3 | Max concurrently executing subtasks |
| `max_agent_attempts` | 3 | Max attempts per agent (actual count decided by the recovery policy) |
| `max_replacement_requests` | 1 | Max replacement requests (replacement-chain depth 1) |
| `max_replan_attempts` | 3 | Max decomposition revisions by the supervisor |
| `sandbox_timeout` | 30 s | Sandbox per-command timeout |
| `sandbox_output_limit` | 1 MiB | Sandbox output cap |
| `scheduler_poll_interval` | 60 s | Daemon polling interval |
| `task_store_path` | `sandbox/task_store.db` | SQLite task store path |

## Capability Status and Roadmap

Implementation status is strictly separated so that vision is never presented as existing capability.

### Implemented

Four-quadrant routing; thin WorkflowEngine and three coordinators; 13-phase shadow FSM; two-stage review (`passed` as the sole approval field + fail closed); authorization proposal with deferred issuance and revoke-on-all-exits; star communication isolation; explicit `reuse_from` reuse; five-layer streaming timeouts; PipelineGate; tool timeout profiles; schema migration; FailureClassifier; failure fingerprints and deduplication; RecoveryPolicy; provider no-replacement rule; FORMAT single bounded contract repair then REPLAN; low-trust agent text cannot forge system failures; SQLite persistence (tasks / Execution / PlanVersion); path-isolated SandboxRunner; MCP protocol layer; semantic-verified dynamic tool registration.

### Deferred (Design Rationale Retained)

- **Multi-dimensional budgets**: a token / cost / time budget system. The existing time caps (see the defaults table) and basic token usage records are single-dimension constraints, not a multi-dimensional budget.
- **Context governance**: context compaction, pruning, and quota management for long tasks.
- **Production container sandbox**: the current SandboxRunner performs path isolation only (see risk statement); container-level isolation (the earlier Docker / remote-execution idea) is deferred.

### Planned (Not Implemented)

- **Unified ModelGateway**: a single gateway layer for model access, quotas, and failover.
- **Execution-level and parameter-level permission hardening**: current grants are tool-name granularity; finer per-parameter constraints are not yet implemented.
- **Temporal PoC or migration**: feasibility validation of replacing in-process orchestration with Temporal.
- **Durable auditing**: cross-task audit trail storage and querying.
- **OpenTelemetry**: distributed tracing and observability integration.
- Other retained future directions: TTL caching of tools with invocation-count-driven persistence thresholds (the earlier Redis idea, replaced today by explicit persistent scheduling), statistics database tables for analysis (the earlier MySQL idea, replaced today by SQLite execution records and local logs), dynamic agent templates and capability whitelists, and a web console with task visualization.

## Risks and Limitations

- **Experimental / research status**: the governance and recovery mechanisms target research and validation and are untested under production load.
- **Streaming cancellation may be delayed underneath**: timeout and cancellation teardown depends on the underlying SDK's cancellation behavior; iterator close has a bounded grace period, and in extreme cases underlying cancellation may be delayed — late results are isolated by the stop gate, but prompt resource release is not guaranteed.
- **The current sandbox is not a security boundary**: SandboxRunner isolates the working directory path only; it does not restrict network, CPU, memory, syscalls, or directory escape, and **must not be used to run untrusted code**.
- **Human escalation is interface and state only**: the `human_required` terminal state produces a complete HumanEscalationRequest handover, but there is no human workstation, notification, or ticket write-back system.
- **Budgets and context deferred**: while multi-dimensional budgets and context governance are deferred, cost and context control for long tasks relies on the time caps and bounded retry quotas above.

## Documentation

- [SO_Agent_Execution_Runtime.md](./SO_Agent_Execution_Runtime.md): English architecture document — original design rationale and actual evolution, layering, state model, data structures, permissions, configuration, and test plan.
- [多_Agent_框架.md](./多_Agent_框架.md): Chinese architecture document (section-for-section and capability-status aligned with the English version).
- Configuration examples: [src/so_agent/api_config.yaml.example](./src/so_agent/api_config.yaml.example) and `src/so_agent/tool_packages/<agent>/api_config.yaml.example`.
