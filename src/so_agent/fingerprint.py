"""SubtaskSpec 指纹计算与兼容判定 — 用于跨计划版本的结果复用校验。

对子任务规格中真正影响执行结果的语义字段做稳定哈希；
subtask_id / plan_version / max_attempts / reuse_from 等元数据不参与
指纹（``reuse_from`` 是声明元数据，若入指纹会使声明本身改变语义
指纹，复用判定自相矛盾）。

任务 35 起，指纹不再作为自动复用决策依据（同名 / 同指纹自动复用
已停用），仅作为主管显式 ``reuse_from`` 声明的安全校验输入：
- ``spec_fingerprint``：无状态哈希，保持历史行为不变；
- ``specs_fingerprint_equivalent``：安全兼容判定 —— 除 allowed_tools
  外的语义字段必须完全一致，依赖集合允许经显式 reuse 声明形成的
  id 映射后集合等价（容忍重规划改名，不容忍任意重构）；
  **绝不用文本相似度放宽**。allowed_tools 不在此判定，因为动态工具
  跨版本可能改名、新计划也允许收缩工具范围，由规划协调器按
  manifest 稳定标识单独校验“不得扩大范围”。
"""

import hashlib
import json

from so_agent.models import SubtaskSpec

__all__ = ["spec_fingerprint", "specs_fingerprint_equivalent"]


def spec_fingerprint(spec: SubtaskSpec) -> str:
    """对 SubtaskSpec 的语义关键字段做 sha256 哈希。

    只包含影响执行结果的字段：instructions / role /
    allowed_tools / expected_output / dependencies。不包含 subtask_id /
    plan_version / max_attempts，也不含 reuse_from（声明元数据）。
    返回 16 位十六进制字符串（64 bit）。
    """
    key_fields = {
        "instructions": spec.instructions,
        "role": spec.role,
        "allowed_tools": sorted(spec.allowed_tools),
        "expected_output": spec.expected_output,
        "dependencies": sorted(spec.dependencies),
    }
    return hashlib.sha256(
        json.dumps(key_fields, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]


def specs_fingerprint_equivalent(
    new_spec: SubtaskSpec,
    old_spec: SubtaskSpec,
    dependency_id_map: dict[str, str] | None = None,
) -> bool:
    """判定两个子任务规格的指纹成分是否安全兼容（显式复用校验用）。

    任务 35 的安全兼容规则（安全优先，不用文本相似度）：

    - ``instructions`` / ``role`` / ``expected_output`` 必须**完全一致**；
    - ``allowed_tools`` **不在本判定内**：动态工具跨版本可能改名、
      新计划也允许收缩工具范围，由规划协调器按 manifest 稳定标识
      单独校验“新 spec 工具范围不得扩大旧 spec”；
    - ``dependencies`` 允许差异，但仅限：旧 spec 的依赖 id 集合经
      ``dependency_id_map``（旧 id → 新 id，仅由本轮合法复用声明构成）
      映射后与新 spec 依赖集合**集合等价**。映射之外的任何改名、
      增删依赖均判不兼容。

    Args:
        new_spec: 新计划子任务规格（携带 reuse_from 声明）。
        old_spec: 旧计划子任务规格（复用来源）。
        dependency_id_map: 本轮合法复用声明形成的旧→新依赖 id 映射；
            None 视为空映射（依赖必须逐字相等）。

    Returns:
        True 表示指纹成分兼容（依赖项还需其余安全校验项通过）；
        False 表示不兼容，显式复用声明应被拒绝。
    """
    if new_spec.instructions != old_spec.instructions:
        return False
    if new_spec.role != old_spec.role:
        return False
    if new_spec.expected_output != old_spec.expected_output:
        return False
    id_map = dependency_id_map or {}
    mapped_old_dependencies = {id_map.get(dep, dep) for dep in old_spec.dependencies}
    return mapped_old_dependencies == set(new_spec.dependencies)
