import os
import json
import pickle
import time
from typing import Dict, Any, List, Optional


class BaseLongTermMemory:
    """Abstract base class for long-term memory storage.

    :param base_dir: 基础目录路径
    :type base_dir: str
    :return: 无
    :rtype: None
    """

    def __init__(self, base_dir: str = "data/long_term") -> None:
        """Initialize the base long-term memory.

        :param base_dir: 基础目录路径
        :type base_dir: str
        :return: 无
        :rtype: None
        """
        self.base_dir = base_dir
        self.knowledge_dir = os.path.join(base_dir, "knowledge")
        self.milestones_dir = os.path.join(base_dir, "milestones")
        self.patterns_dir = os.path.join(base_dir, "patterns")
        os.makedirs(self.knowledge_dir, exist_ok=True)
        os.makedirs(self.milestones_dir, exist_ok=True)
        os.makedirs(self.patterns_dir, exist_ok=True)

    def save_knowledge(self, key: str, data: Dict[str, Any]) -> str:
        """Save knowledge data to a JSON file.

        :param key: 知识键名
        :type key: str
        :param data: 知识数据
        :type data: Dict[str, Any]
        :return: 保存的文件路径
        :rtype: str
        """
        filepath = os.path.join(self.knowledge_dir, f"{key}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return filepath

    def load_knowledge(self, key: str) -> Optional[Dict[str, Any]]:
        """Load knowledge data from a JSON file.

        :param key: 知识键名
        :type key: str
        :return: 知识数据或 None
        :rtype: Optional[Dict[str, Any]]
        """
        filepath = os.path.join(self.knowledge_dir, f"{key}.json")
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def list_knowledge_keys(self) -> List[str]:
        """List all knowledge keys.

        :param: 无
        :type: 无
        :return: 知识键名列表
        :rtype: List[str]
        """
        files = os.listdir(self.knowledge_dir)
        return [f.replace('.json', '') for f in files if f.endswith('.json')]

    def save_milestone(self, milestone: Dict[str, Any]) -> str:
        """Save a milestone to a JSON file.

        :param milestone: 里程碑数据
        :type milestone: Dict[str, Any]
        :return: 保存的文件路径
        :rtype: str
        """
        timestamp = time.time()
        filename = f"milestone_{timestamp:.6f}.json"
        filepath = os.path.join(self.milestones_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(milestone, f, ensure_ascii=False, indent=2)
        return filepath

    def load_milestones(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Load recent milestones.

        :param limit: 加载数量限制
        :type limit: int
        :return: 里程碑列表
        :rtype: List[Dict[str, Any]]
        """
        files = sorted(os.listdir(self.milestones_dir), reverse=True)
        milestones = []
        for filename in files[:limit]:
            if filename.endswith('.json'):
                filepath = os.path.join(self.milestones_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    milestones.append(json.load(f))
        return milestones

    def save_pattern(self, pattern_id: str, pattern: Dict[str, Any]) -> str:
        """Save a pattern to a JSON file.

        :param pattern_id: 模式 ID
        :type pattern_id: str
        :param pattern: 模式数据
        :type pattern: Dict[str, Any]
        :return: 保存的文件路径
        :rtype: str
        """
        filepath = os.path.join(self.patterns_dir, f"{pattern_id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(pattern, f, ensure_ascii=False, indent=2)
        return filepath

    def load_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Load a pattern from a JSON file.

        :param pattern_id: 模式 ID
        :type pattern_id: str
        :return: 模式数据或 None
        :rtype: Optional[Dict[str, Any]]
        """
        filepath = os.path.join(self.patterns_dir, f"{pattern_id}.json")
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_object(self, key: str, obj: Any) -> str:
        """Save an object as a pickle file.

        :param key: 对象键名
        :type key: str
        :param obj: 要保存的对象
        :type obj: Any
        :return: 保存的文件路径
        :rtype: str
        """
        filepath = os.path.join(self.base_dir, f"{key}.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        return filepath

    def load_object(self, key: str) -> Optional[Any]:
        """Load an object from a pickle file.

        :param key: 对象键名
        :type key: str
        :return: 加载的对象或 None
        :rtype: Optional[Any]
        """
        filepath = os.path.join(self.base_dir, f"{key}.pkl")
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def delete(self, key: str) -> bool:
        """Delete a knowledge entry or object by key.

        :param key: 键名
        :type key: str
        :return: 是否成功删除
        :rtype: bool
        """
        for ext in ['.json', '.pkl']:
            filepath = os.path.join(self.base_dir, f"{key}{ext}")
            if os.path.exists(filepath):
                os.remove(filepath)
                return True
        for subdir in [self.knowledge_dir, self.milestones_dir, self.patterns_dir]:
            for ext in ['.json', '.pkl']:
                filepath = os.path.join(subdir, f"{key}{ext}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                    return True
        return False

    def get_storage_stats(self) -> Dict[str, Any]:
        """Return storage statistics for each sub-directory.

        :param: 无
        :type: 无
        :return: 存储统计信息
        :rtype: Dict[str, Any]
        """
        stats = {}
        for subdir_name, subdir_path in [
            ('knowledge', self.knowledge_dir),
            ('milestones', self.milestones_dir),
            ('patterns', self.patterns_dir),
        ]:
            count = 0
            total_size = 0
            for filename in os.listdir(subdir_path):
                filepath = os.path.join(subdir_path, filename)
                if os.path.isfile(filepath):
                    count += 1
                    total_size += os.path.getsize(filepath)
            stats[subdir_name] = {
                'file_count': count,
                'total_size_kb': round(total_size / 1024, 2),
            }
        return stats


class LongTermMemory(BaseLongTermMemory):
    """Concrete long-term memory implementation.

    :param: 无
    :type: 无
    :return: 无
    :rtype: None
    """
    pass
