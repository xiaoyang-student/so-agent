"""四象限任务路径（paths）。

按任务类型划分的执行路径：
- ① direct_answer：简单+短 —— 直接一次 LLM 调用回答；
- ② deep_reasoning：复杂+短 —— 多轮推理后回答；
- ③ persistent_task：简单+长 —— 持久化状态 + 调度执行；
- ④ full_workflow：复杂+长 —— 走完整编排工作流（WorkflowEngine）。

各路径对外统一暴露 ``run(objective, constraints, acceptance_criteria,
classification) -> OrchestratorOutput`` 异步接口，由主管路由器按
``ClassificationResult.task_type`` 分发。

说明：为避免循环导入以及尚未落地的路径模块导致整个包导入失败，
本包不在 ``__init__`` 中导入任何子模块；使用方按需直接导入具体
路径模块（例如 ``from so_agent.paths.persistent_task import
PersistentTaskPath``）。
"""
