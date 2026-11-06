"""工具包集合。

包含主 Agent 的三个固定工具入口与动态生成工具目录：
- code_agent：代码执行 Agent；
- subagent_creator：按需动态创建专用子 Agent 的 Agent；
- review_agent：评审与结果校验 Agent；
- generated_tools：由 subagent_creator 产出、经评审通过后注册的动态工具。

每个 Agent 工具包目录遵循统一结构：agent.py（构建逻辑）、
prompt.md（系统提示词）、api_config.yaml（模型与连接配置，密钥只存环境变量名）。
"""
