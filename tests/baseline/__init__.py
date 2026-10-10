"""so-agent 基准场景测试基座（任务 44）。

目录约定：
- fakes.py：无真实模型调用的共享假件（假 Runner、剧本主管、评审脚本、
  引擎组装入口），从既有集成测试（test_workflow / test_replan_reuse /
  test_workflow_timeout）的成熟模式提取归一；
- scenarios.py：基准场景工厂——简单问答、DAG、网络失败、超时、取消、
  重规划与显式复用（reuse），供本阶段冒烟验证与后续阶段的回归基线
  复用；本阶段只提供可复用入口，不执行任何真实付费模型请求；
- test_baseline_scenarios.py：基座冒烟测试——每个场景跑一遍并断言
  核心终态行为（避免脆弱的全量快照断言）。
"""
