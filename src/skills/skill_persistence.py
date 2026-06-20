"""技能持久化与跨任务迁移。

将技能库保存到 LongTermMemory 并在新会话中加载，使 Agent 能跨任务、
跨会话积累技能知识，实现真正的自进化。

第一性原理：
- 技能是 Agent 的核心知识资产，不应随会话结束而丢失
- 技能由元数据（名称/能力/统计）与可执行 action 两部分组成
- 元数据始终可序列化为 JSON；action 可通过 pickle 持久化（命名函数可行，
  lambda 不可序列化时降级为元数据模式，加载后 action 为 None）
- 加载时优先尝试完整恢复，失败则回退到元数据模式，保证韧性
"""

import logging
import pickle
from typing import Any, Dict, List, Optional

from src.memory.long_term import BaseLongTermMemory

from .skill import Skill
from .skill_registry import SkillRegistry

logger = logging.getLogger(__name__)


class SkillPersistence:
    """技能持久化器：在 LongTermMemory 中保存与加载技能。

    :param memory: 长期记忆存储后端
    """

    SKILL_META_PREFIX = "skill_meta_"
    SKILL_OBJ_PREFIX = "skill_obj_"

    def __init__(self, memory: BaseLongTermMemory) -> None:
        self.memory = memory

    def save_skill(self, skill: Skill) -> Dict[str, Any]:
        """保存单个技能。

        :return: 保存结果摘要，包含 ``name``、``metadata_saved``、``action_saved``
        """
        meta = skill.to_dict()
        self.memory.save_knowledge(f"{self.SKILL_META_PREFIX}{skill.name}", meta)

        obj_saved = False
        try:
            self.memory.save_object(f"{self.SKILL_OBJ_PREFIX}{skill.name}", skill)
            obj_saved = True
        except (TypeError, pickle.PickleError, AttributeError, ValueError):
            pass

        return {
            'name': skill.name,
            'metadata_saved': True,
            'action_saved': obj_saved,
        }

    def save_registry(self, registry: SkillRegistry) -> List[Dict[str, Any]]:
        """保存注册中心中的全部技能。"""
        results: List[Dict[str, Any]] = []
        for name in registry.list_skills():
            skill = registry.get(name)
            if skill is not None:
                results.append(self.save_skill(skill))
        return results

    def load_skill(self, name: str) -> Optional[Skill]:
        """加载单个技能。

        优先尝试完整加载（含 action），失败则回退到元数据模式（action=None）。
        """
        try:
            obj = self.memory.load_object(f"{self.SKILL_OBJ_PREFIX}{name}")
            if isinstance(obj, Skill):
                return obj
        except Exception as e:
            logger.debug("SkillPersistence.load_skill: full load failed for %s, fall back to metadata: %s", name, e)

        meta = self.memory.load_knowledge(f"{self.SKILL_META_PREFIX}{name}")
        if meta is None:
            return None
        return Skill(
            name=meta.get('name', name),
            description=meta.get('description', ''),
            capabilities=meta.get('capabilities', []),
            action=None,
        )

    def load_into(
        self,
        registry: SkillRegistry,
        overwrite: bool = False,
    ) -> List[str]:
        """将持久化的技能加载到注册中心。

        :param registry: 目标注册中心
        :param overwrite: 同名技能是否覆盖
        :return: 成功加载的技能名列表
        """
        loaded: List[str] = []
        existing = set(registry.list_skills())
        for name in self.list_persisted():
            if name in existing and not overwrite:
                continue
            skill = self.load_skill(name)
            if skill is None:
                continue
            if name in existing:
                registry.unregister(name)
            registry.register(skill)
            loaded.append(name)
        return loaded

    def list_persisted(self) -> List[str]:
        """列出已持久化的技能名。"""
        names: List[str] = []
        for key in self.memory.list_knowledge_keys():
            if key.startswith(self.SKILL_META_PREFIX):
                names.append(key[len(self.SKILL_META_PREFIX):])
        return names

    def delete_skill(self, name: str) -> bool:
        """删除已持久化的技能。"""
        meta_deleted = self.memory.delete(f"{self.SKILL_META_PREFIX}{name}")
        obj_deleted = self.memory.delete(f"{self.SKILL_OBJ_PREFIX}{name}")
        return meta_deleted or obj_deleted
