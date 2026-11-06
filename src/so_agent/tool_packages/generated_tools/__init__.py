"""动态生成工具目录。

存放由子 Agent 创建器（subagent_creator）产出、经评审 Agent 评审通过后
注册为可用工具的代码与清单：

- registry.json：工具清单（数组，元素结构见 models.GeneratedToolManifest），
  记录每个工具的入口文件、描述、输入输出 schema、评审状态与版本；
- 每个工具以独立子目录或模块文件形式存放，入口文件相对本目录定位。

运行时只允许加载 review_status 为 approved 的工具。
"""
