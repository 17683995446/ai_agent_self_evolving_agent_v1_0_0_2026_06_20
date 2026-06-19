import time
import uuid
from typing import Any, Dict, List, Optional

from src.memory.long_term import LongTermMemory
from src.skills.skill_learner import SkillLearner
from src.skills.skill_registry import SkillRegistry


class LongTaskReviewer:
    """长任务离线复盘引擎：对长任务执行轨迹进行持久化与复盘分析。

    - 将 mission summary 持久化到长期记忆，形成可追溯的执行轨迹。
    - 对失败步骤进行归因分析，输出技能优化建议。
    - 通过 SkillLearner 将复盘结论反馈到技能注册表，驱动技能自增强。
    - 生成结构化复盘报告，支持后续策略进化。
    """

    def __init__(
        self,
        memory: LongTermMemory,
        skill_registry: Optional[SkillRegistry] = None,
        skill_learner: Optional[SkillLearner] = None,
    ) -> None:
        self.memory = memory
        self.skill_registry = skill_registry or SkillRegistry()
        self.skill_learner = skill_learner or SkillLearner(self.skill_registry)
        self._reviews: List[Dict[str, Any]] = []

    def persist_mission(self, summary: Dict[str, Any]) -> str:
        """将 mission 执行摘要持久化到长期记忆。"""
        review_id = f"mission_{summary.get('plan_id', uuid.uuid4().hex[:8])}"
        record = {
            'review_id': review_id,
            'summary': summary,
            'persisted_at': time.time(),
        }
        self.memory.save_knowledge(review_id, record)
        return review_id

    def review(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """对单次 mission 执行结果进行复盘分析。

        - 统计成功/失败步骤分布。
        - 归因失败步骤到具体技能，输出改进建议。
        - 通过 SkillLearner 吸收执行反馈。
        - 生成结构化复盘报告并持久化。
        """
        start = time.time()
        results: List[Dict[str, Any]] = summary.get('results', [])
        total = len(results)
        success_count = sum(1 for r in results if r.get('success'))
        failure_count = total - success_count

        # 失败步骤归因
        failure_attribution: List[Dict[str, Any]] = []
        for record in results:
            if not record.get('success'):
                failure_attribution.append({
                    'task_name': record.get('task_name'),
                    'task_id': record.get('task_id'),
                    'worker_id': record.get('worker_id'),
                    'error': record.get('result'),
                })
            # 统一吸收反馈到学习器
            self.skill_learner.learn_from_record(record)

        # 技能优化建议
        recommendations = self.skill_learner.improve()

        report = {
            'review_id': f"review_{uuid.uuid4().hex[:8]}",
            'plan_id': summary.get('plan_id'),
            'status': summary.get('status'),
            'total_steps': total,
            'success_steps': success_count,
            'failure_steps': failure_count,
            'success_rate': success_count / total if total > 0 else 0.0,
            'failure_attribution': failure_attribution,
            'skill_recommendations': recommendations,
            'duration': time.time() - start,
            'timestamp': time.time(),
        }

        # 持久化复盘报告
        self.memory.save_knowledge(report['review_id'], report)
        self._reviews.append(report)
        return report

    def review_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """返回最近的复盘记录。"""
        return list(self._reviews[-limit:])

    def generate_report(self, review_result: Dict[str, Any]) -> str:
        """生成人类可读的复盘报告。"""
        lines = [
            "=" * 50,
            "Long Task Review Report",
            "=" * 50,
            f"Review ID: {review_result.get('review_id', 'N/A')}",
            f"Plan ID: {review_result.get('plan_id', 'N/A')}",
            f"Status: {review_result.get('status', 'unknown')}",
            f"Total Steps: {review_result.get('total_steps', 0)}",
            f"Success Steps: {review_result.get('success_steps', 0)}",
            f"Failure Steps: {review_result.get('failure_steps', 0)}",
            f"Success Rate: {review_result.get('success_rate', 0.0):.2%}",
            f"Duration: {review_result.get('duration', 0.0):.3f}s",
            "",
            "Failure Attribution:",
        ]
        for fa in review_result.get('failure_attribution', []):
            lines.append(f"  - task={fa.get('task_name')} worker={fa.get('worker_id')} error={fa.get('error')}")
        lines.append("")
        lines.append("Skill Recommendations:")
        recs = review_result.get('skill_recommendations', {})
        for category, skills in recs.items():
            lines.append(f"  {category}: {', '.join(skills) if skills else '(none)'}")
        lines.append("=" * 50)
        return "\n".join(lines)

    def load_review(self, review_id: str) -> Optional[Dict[str, Any]]:
        """从长期记忆加载指定复盘记录。"""
        return self.memory.load_knowledge(review_id)
