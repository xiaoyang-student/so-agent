# Multi-Agent Framework Structure Blueprint

## I. Design Goals

- The Supervisor Agent owns all tools; its tool catalog is strictly limited to the Code Agent, the Subagent Creator, and the Review Agent.
- The Supervisor Agent is responsible for understanding goals, decomposing tasks, granting tools, dispatching tasks, and organizing the final answer; it does not directly access the file system or launch processes.
- Subagents have no tool capabilities by default; when dispatching a task, the Supervisor Agent specifies the subset of tools allowed for that task, and unauthorized tools are neither visible nor callable.
- Subagents are forbidden from communicating directly, sharing sessions, or invoking one another; all inputs, results, and feedback must pass through the Supervisor Agent.
- Dedicated subagents are created at runtime and released after use, avoiding a large set of predefined fixed agents.
- Review covers task decomposition, authorization scope, and final results, and uses a layered failure-escalation mechanism to bound retries.
- Each subagent may attempt execution at most three times; after the original subagent fails, it may request a replacement agent from the Supervisor Agent only once.
- A replacement agent cannot request another replacement agent; if it still fails, a task-decomposition review is triggered.
- The Supervisor Agent may revise the task decomposition at most three times; once the quota is exhausted, the system enters the human-escalation state; the first version only retains the interface and state, without implementing a human-handling system.
- The first-version sandbox is only for validating the toolchain: it writes code into an isolated directory and runs it through a subprocess, without providing true security isolation.

## II. Overall Layering

### Orchestration Layer

The Supervisor Agent sits at the top layer, receives user goals, and produces subtask plans. It exclusively owns the tool catalog and decides whether each subagent may use tools and which tools it may use. The Supervisor Agent itself registers only the following three agent tools:

1. Code Agent
   - Has its own project package, prompt file, and API configuration file.
   - Creates, modifies, and inspects code files.
   - Invokes the internal sandbox runner to verify code when necessary.
   - Writes newly generated tools into the dedicated generated-tools package, never into other agent packages.
   - Returns a change summary, run results, and error information.

2. Subagent Creator
   - Has its own project package, prompt file, and API configuration file.
   - Receives subtask definitions and tool-authorization lists issued by the Supervisor Agent.
   - Dynamically constructs dedicated agents based on role, prompt, inputs, and authorized scope.
   - Subagents have an empty default tool set, mounting only tools explicitly approved by the Supervisor Agent.
   - Schedules dependency-free subtasks for concurrent execution and advances other tasks serially according to dependencies.
   - Exposes no references, sessions, or message channels of other agents to any subagent.
   - Returns each subagent's structured result separately to the Supervisor Agent, which organizes the aggregation.
   - The aggregation agent shown in the diagram is an on-demand role that receives only result copies provided by the Supervisor Agent, does not communicate with other subagents, and is not a fourth fixed tool.

3. Review Agent
   - Has its own project package, prompt file, and API configuration file.
   - Before execution, checks whether the task decomposition is complete, executable, and aligned with the user's goal.
   - Checks whether the tools requested by each subtask are necessary, so that the Supervisor Agent can make authorization decisions.
   - After execution, checks the correctness, completeness, evidence, and risks of the results.
   - Outputs pass or reject, an issue list, required fixes, and rework targets.
   - The evaluation agent shown in the diagram is merged with the Review Agent into a single capability, adding no new tool.

### Control Layer

The control layer is implemented in plain Python code and is not exposed as a model-visible tool:

- State machine: constrains tasks to flow only through the defined phases.
- Scheduler: handles dependencies, concurrency limits, timeouts, and cancellation.
- Agent registry: used only by the Supervisor Agent and the control layer; records the identity, capabilities, status, and lifecycle of dynamic agents.
- Tool authorization gateway: validates the caller, task ID, tool name, authorization scope, and validity period, and rejects all unauthorized calls.
- Supervisor router: permits only vertical communication between the Supervisor Agent and each subagent, and forbids horizontal message passing between subagents.
- Retry controller: bounds review-rework counts to prevent infinite loops.
- Event recorder: stores task dispatch, authorization, tool calls, outputs, durations, and errors.

### Infrastructure Layer

These are internal implementation services and do not count toward the three tools of the main agent:

- Model client.
- File read/write adapters.
- Subprocess runner.
- Sandbox directory manager.
- Configuration and logging components.

## III. Complete Execution Flow

1. Receive task
   - Create a task ID.
   - Fix the user goal, constraints, and acceptance criteria.

2. Supervisor Agent decomposition and authorization
   - Produce a subtask list.
   - Specify the purpose, inputs, dependencies, role, and completion conditions for each subtask.
   - Select the subset of tools allowed for the subtask from the three tools; no tools are authorized by default.
   - Generate an authorization record for each dispatch that subagents cannot modify.

3. First-round review
   - The Supervisor Agent invokes the Review Agent to check the decomposition plan and whether tool authorizations are necessary and minimal.
   - If rejected, re-decompose or tighten authorization based on the feedback.
   - If rejected three consecutive times, terminate and return an explicit reason.

4. Execute tasks
   - The Supervisor Agent packages the task content, context copy, and allowed-tool list before dispatching.
   - Coding tasks may be handed to the Code Agent.
   - Research, analysis, aggregation, and similar tasks may be handed to the Subagent Creator, which dynamically creates dedicated agents.
   - Subagents may call only the tools explicitly authorized for the current task; unauthorized tools are neither injected into their tool sets nor accepted by the authorization gateway.
   - The scheduler may execute mutually independent subtasks concurrently, but subagents are mutually invisible and cannot communicate.
   - Each execution result can only be returned to the Supervisor Agent and cannot be sent directly to other subagents.

5. Aggregate results
   - The Supervisor Agent collects the independent results of each subagent.
   - If an aggregation role is needed, the Supervisor Agent creates it through the Subagent Creator and provides it only with the necessary result copies and explicit authorization.
   - The aggregation role does not obtain identity references or communication capabilities of other subagents.
   - The aggregated result must preserve the source, execution status, and evidence corresponding to each conclusion.

6. Final review
   - The Review Agent checks the results against the original goal and acceptance criteria.
   - If passed, the Supervisor Agent outputs the final result.
   - If rejected, only the flagged tasks are reworked; tasks that already passed are not re-executed.

### Subagent Failure Escalation Flow

Each task-decomposition version follows the fixed flow below:

1. Original subagent execution
   - A single subagent is executed at most three times.
   - After three attempts without meeting the completion conditions, it may send one ReplacementRequest to the Supervisor Agent.
   - This request may be sent only once; duplicate requests are rejected directly by the control layer.

2. Supervisor Agent handles the replacement request
   - The Supervisor Agent checks the failure evidence, tools used, and task authorization.
   - If approved, it generates a new replacement agent through the Subagent Creator and re-dispatches the same subtask.
   - The Supervisor Agent may adjust the replacement agent's prompt and authorized tools, but must not change the original main task.

3. Replacement agent execution
   - The replacement agent is likewise executed at most three times.
   - The replacement agent's can_request_replacement is fixed to false, and it cannot request another new agent.
   - If the replacement agent still fails, it can only return a TaskDecompositionIssue to the Supervisor Agent, explaining that the task decomposition may have problems, together with the failure evidence.

4. Supervisor Agent re-decomposition
   - After receiving the TaskDecompositionIssue, the Supervisor Agent re-examines and revises the task decomposition.
   - The Supervisor Agent has at most three decomposition-revision opportunities; each produces a new plan version that again goes through review by the Review Agent.
   - Each new plan version recreates the original subagent; that original subagent still has one replacement-request entitlement, but its replacement agent still has none.

5. Human escalation
   - If the task still cannot be completed after the third task-decomposition revision, the state transitions to human_required.
   - The control layer generates a HumanEscalationRequest containing the original task, all plan versions, failure records, tool-call evidence, and the final review opinion.
   - The first version only defines the human-escalation interface, events, and state; it does not implement a human workstation, notifications, or a handling process.

This mechanism simultaneously bounds single-agent retries, replacement-chain depth, and supervisor re-planning counts, avoiding unbounded agent creation or infinite loops.

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

## V. Core Data Structures

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

All inter-agent communication uses these structured objects, avoiding reliance on free text to guess state.

## VI. Permission Design

- The Supervisor Agent holds the ownership, registration, assignment, and revocation rights for the three agent tools.
- Subagents have no tool permissions by default; the Supervisor Agent must explicitly specify the allowed-tool list on every task dispatch.
- Tool permissions are granted per task; they are not automatically inherited by subsequent tasks and cannot be re-delegated by subagents.
- When creating a subagent, only the approved tools are injected into that agent's tool set; the authorization gateway performs one more validation at runtime.
- When a subagent requests additional tools, it can only end the current execution and return the request rationale to the Supervisor Agent; it cannot call, discover, or register tools on its own.
- After three consecutive failures, the original subagent may request a replacement agent from the Supervisor Agent only once; this entitlement is non-transferable and non-reusable.
- A replacement agent has its replacement-request capability removed at creation time and cannot extend the replacement chain further.
- Subagents have no message channels, shared sessions, or mutual references; replacement requests, failure reports, and result exchanges must all be sent to the Supervisor Agent.
- The Code Agent may use its internal project-directory read/write capability and sandbox execution capability only after authorization by the Supervisor Agent.
- New tools generated by the Code Agent are stored only as candidate code; they are never automatically registered, automatically executed, or automatically granted to any subagent.
- Even after a candidate tool passes review, the Supervisor Agent still decides whether to register it and in which specific task to authorize its use.
- The Review Agent runs only after authorization by the Supervisor Agent and remains read-only, without modifying artifacts.
- The Subagent Creator may create agents only according to the tasks and authorizations issued by the Supervisor Agent; it cannot expand permissions, bypass the registry, or exceed concurrency limits.
- Every subtask and every tool call must be traceable to the main task, the authorization record, and the Supervisor Agent; the main task may not be changed dynamically.

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

- Use asyncio.create_subprocess_exec to create an independent Python process, preferred over threads.
- Use the Python interpreter of the current virtual environment.
- Pin the process working directory to the current run directory.
- Set a run timeout and terminate the process after it expires.
- Capture standard output, standard error, exit code, and duration.
- Use a fresh directory for each run to prevent different tasks from overwriting each other.
- Explicitly mark that this approach provides only path isolation and does not restrict network, CPU, memory, system calls, or directory escape; therefore, it must not execute untrusted code.

## VIII. Recommended Project Structure

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
          <tool_name>/
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

The tool-class package fixedly contains four project packages:

1. code_agent
   - agent.py defines the Code Agent and its internal capabilities.
   - prompt.md stores the Code Agent prompt separately.
   - api_config.yaml stores the model, endpoint URL, timeout, and retry configuration for this agent.

2. subagent_creator
   - agent.py defines the subagent creation and scheduling logic.
   - prompt.md stores the creator prompt separately.
   - api_config.yaml stores this agent's independent model-call configuration.

3. review_agent
   - agent.py defines the decomposition review, authorization review, and result review logic.
   - prompt.md stores the review prompt separately.
   - api_config.yaml stores this agent's independent model-call configuration.

4. generated_tools
   - Dedicated to storing tools generated by the Code Agent; each tool uses its own subdirectory.
   - tool.py is the tool implementation, manifest.yaml describes the name, version, inputs/outputs, and source task, and test_tool.py stores minimal verification tests.
   - registry.json records only candidate tools and their review and registration status.
   - This directory is a tool code repository, not a fourth tool entry for the Supervisor Agent.

Configuration security rules:

- API configuration files store only provider, model, base_url, timeout, retry, and the name of the key environment variable.
- API keys are not written into the repository; they are injected through environment variables.
- The three agents may use different models or service endpoints, but all are parsed and validated by the unified config_loader.py.

Core file responsibilities:

- orchestrator.py: defines the Supervisor Agent, which holds exactly and only the three agent tool entries.
- workflow.py: implements the state machine, two-stage review, and at most three reworks.
- models.py: stores all structured input/output models.
- context.py: stores the task context, registry, sandbox paths, and event-recorder references.
- registry.py: the Supervisor Agent manages the creation, query, count limit, and release of dynamic agents; no query interface is exposed to subagents.
- scheduler.py: handles the dependency graph, isolated concurrent execution, three attempts per agent, one replacement request, and failure propagation.
- sandbox.py: writes code files and launches subprocesses.
- tool_registry.py: stores the three agent tools and the metadata of approved generated tools; ownership belongs to the Supervisor Agent.
- permissions.py: issues, validates, and revokes task-level tool authorizations.
- supervisor_router.py: forces all agent messages, replacement requests, and task-decomposition issues through the Supervisor Agent, and rejects inter-subagent communication.
- config_loader.py: reads the API configurations from the three agent packages separately, without reading or persisting plaintext keys.
- Human escalation retains only the HumanEscalationHandler interface and the human_required state in workflow.py, without implementing an external human system.

## IX. SDK Usage Boundaries

- Use the Python OpenAI Agents SDK.
- The Supervisor Agent holds the sole registration entry for the three agent tools through Agent.as_tool.
- The three fixed agents each load prompts and API configuration from their own package, avoiding coupling between responsibilities and model parameters.
- Dynamic roles are constructed at runtime by the Subagent Creator with an empty default tools list; only tools approved in a ToolGrant can be injected.
- Tools generated by the Code Agent are first written into generated_tools and enter the authorization candidate set only after passing review and being registered by the Supervisor Agent.
- Do not configure handoffs for subagents or share sessions, avoiding any horizontal communication path.
- The Review Agent uses Pydantic models to output ReviewDecision.
- The Runner is responsible for a single agent execution, while the outer workflow.py handles authorization, routing, and business retries; the two responsibilities are separated.
- Explicitly set the maximum turns, model timeout, and tool timeout for each run.
- Install dependencies in the project's dedicated virtual environment; avoid directly upgrading the openai package in the current global environment.

## X. Configuration Recommendations

- Maximum number of dynamic agents: 8 by default.
- Maximum concurrency: 3 by default.
- Default tool count for subagents: 0.
- Tool authorization validity: limited to a single subtask execution.
- Subagent horizontal communication: always disabled.
- Execution attempts per subagent task: at most 3.
- Replacement requests by the original subagent: at most 1.
- Replacement agent requesting again: disabled.
- Task-decomposition revisions by the Supervisor Agent: at most 3.
- Decomposition review retries: counted within the three task-decomposition revision quota.
- Result review rework: enters the failure-escalation flow of the corresponding subtask.
- Maximum model turns per agent: 10 by default.
- Model call timeout: 60 seconds by default.
- Sandbox process timeout: 30 seconds by default.
- Sandbox output limit: 1 MB by default; excess output is truncated and recorded.

All the above parameters can be overridden through configuration files or environment variables and are not hard-coded in prompts.

## XI. Test Plan

- Unit tests: state transitions, dependency scheduling, authorization issuance and revocation, registry limits, sandbox timeouts, and count limits at each level.
- Package-structure tests: confirm that the tool-class package fixedly contains three agent packages and one generated-tools package, and that all three agent packages have prompts and API configuration.
- Configuration tests: confirm that the three API configurations can be loaded independently, that missing fields are rejected, and that no plaintext keys exist in the repository.
- Main-configuration tests: confirm that the Supervisor Agent's tool set strictly equals the three designated tools and that the default tool set of subagents is empty.
- Generated-tool tests: confirm that the Code Agent can write only into generated_tools and that candidate tools are not callable before review and supervisor registration.
- Permission tests: confirm that subagents can call only tools approved for the current task, cannot re-delegate permissions, and that expired or over-privileged calls always fail.
- Communication-isolation tests: confirm that subagents cannot obtain references to one another, share sessions, or send messages directly.
- Replacement-flow tests: confirm that the original subagent can request a replacement only once after three failures, that the replacement agent has no request capability, and that after failing it can only report a task-decomposition issue.
- Re-planning tests: confirm that the Supervisor Agent may revise the decomposition at most three times, and that on the fourth time it does not continue running but instead generates a HumanEscalationRequest.
- Human-escalation tests: confirm that only the human_required state and complete handover data are produced, without invoking an unimplemented human system.
- Simulated integration tests: use fake models to verify decomposition, authorization, dynamic creation, replacement, re-planning, aggregation, rejection, and human-escalation flows, without consuming the real API.
- Sandbox tests: verify code writing, subprocess execution, output capture, failure exit, and timeout termination.
- Real end-to-end tests: after an API key is provided, execute one complete task and confirm that the final result has gone through two-stage review.

## XII. Phased Rollout Recommendations

### Minimal Runnable Version

- Set up the project, the virtual environment, and the four tool project packages.
- Create prompt files and API configuration files for each of the three fixed agents.
- Define the structured models and the runtime context.
- Implement the three fixed agent tools and the generated-tools repository.
- Implement the dynamic agent registry.
- Implement the validation sandbox.
- Implement two-stage review, three subagent attempts, one replacement request, and three supervisor re-plannings.
- Define the human-escalation interface and state, without implementing the human-handling endpoint.
- Provide a command-line entry point and simulated tests.

### Stability Enhancements

- Add persistent event logs, task recovery, and cancellation mechanisms.
- Add token, duration, and concurrency budgets.
- Add dynamic agent templates and a capability whitelist.

### Production Enhancements

- Replace the validation sandbox with Docker or a remote isolated execution environment.
- Add observability, auditing, key management, and multi-tenant isolation.
- Add a web console and task visualization.

## XIII. Key Assumptions and Risks

- The workspace is currently empty and can be designed as a greenfield project, without considering legacy-code compatibility.
- The first version uses a dedicated Python virtual environment.
- The code execution directory is not a secure sandbox and is only suitable for trusted test code.
- Dynamic agents do not automatically become new tools of the Supervisor Agent; creation and execution are completed inside the Subagent Creator.
- Code in generated_tools is merely candidate tools and will not enter the callable scope without review and registration by the Supervisor Agent.
- The three API configurations are mutually independent, but keys are uniformly read only from environment variables.
- The Supervisor Agent is the sole tool owner and message hub; subagents cannot expand their own permissions or communicate with each other.
- Each plan version contains at most one layer of replacement agent; a replacement agent is not allowed to request further replacements.
- The Supervisor Agent may revise the task decomposition at most three times, after which it must enter the human-escalation state rather than continue automatic retries.
- Human escalation in the first version is only a logical placeholder, with no UI, notifications, tickets, or human write-back capability.
- Model outputs may still fail even with structured types; the control layer must catch parsing errors.
- Model calls incur cost and uncertainty; tests should prefer fake models.
