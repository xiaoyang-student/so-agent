"""so-agent：多 Agent 编排框架。

主 Agent（Supervisor）拥有并仅拥有三个工具入口：
- code_agent：代码执行 Agent；
- subagent_creator：用于按需动态创建专用子 Agent 的 Agent；
- review_agent：评审与结果校验 Agent。

任务拆分后的执行类、汇总类子 Agent 均通过 subagent_creator 动态产生，
不预置为工具入口。
"""

__version__ = "0.1.0"
