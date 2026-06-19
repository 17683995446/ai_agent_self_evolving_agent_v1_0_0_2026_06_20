"""阶段114：超大规模复杂场景压力测试

目的：验证阶段113静态核验发现的严重bug在真实运行中是否触发。
每项测试都对应一个已发现的严重/高优先级bug，通过真实运行场景复现。

测试覆盖的bug清单：
1. short_term.py 缓冲区刷盘机制整体失效（3个CRITICAL）
   - _should_flush触发时直接插入而非刷盘，缓冲区数据永久卡住
   - _flush_all finally块无条件清空缓冲区，flush失败时数据丢失
   - _handle_flush_error恢复后不重试flush，数据静默丢失
2. agent_loop.py _cleanup_resources无try/finally，异常路径资源泄漏
3. task_pool.py 新任务因score=0被_enforce_size_limit立即驱逐
4. snapshot_manager.py 负max_count导致min([])崩溃
5. auto_rollback.py 无退化检测窗口，单次毛刺即触发回滚
6. micro_optimizer.py level=0.0误判 + 平滑器不学习
"""
import os
import sys
import time
import tempfile
import threading
import sqlite3
import shutil
import logging
from unittest.mock import MagicMock, patch

# 添加项目根目录到路径（源码用 from src.xxx 导入）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("stage114")


def section(title):
    logger.info("=" * 70)
    logger.info("TEST SECTION: %s", title)
    logger.info("=" * 70)


def result(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    logger.info("[%s] %s -- %s", status, name, detail)
    return passed


# =============================================================================
# Bug 1: short_term.py 缓冲区刷盘机制失效（3个CRITICAL）
# =============================================================================

def test_bug1a_should_flush_triggers_direct_insert_loses_buffer():
    """CRITICAL: _should_flush触发时直接插入当前条，但缓冲区已有数据未刷盘。

    场景：缓冲区已有batch_size-1条数据，再写入1条触发_should_flush，
    此时_should_flush返回True，store_sample直接调用_insert_sample插入当前条，
    但缓冲区里之前的batch_size-1条数据完全没有被刷盘，永久卡在缓冲区。
    """
    section("Bug1a: _should_flush触发时缓冲区数据永久卡住")
    from src.memory.short_term import ShortTermMemory

    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        mem = ShortTermMemory(db_path=db_path, batch_size=5, flush_interval=9999)

        # 先写入4条（不触发刷盘，因为4 < 5）
        for i in range(4):
            sample = {
                'timestamp': time.time(),
                'state': {'cpu': float(i)},
                'prediction': {'cpu': float(i)},
                'actual': {'cpu': float(i)},
                'error': {'cpu': 0.1},
                'task_id': f'task_{i}',
                'reward': 0.5,
                'value_score': 0.5,
            }
            ret = mem.store_sample(sample)
            logger.info("写入第%d条，返回=%s, 缓冲区大小=%d", i + 1, ret, len(mem._sample_buffer))

        # 此时缓冲区应有4条
        buffer_count_before = len(mem._sample_buffer)
        db_count_before = mem.get_sample_count()
        logger.info("触发前：缓冲区=%d, 数据库=%d", buffer_count_before, db_count_before)

        # 写入第5条，应触发_should_flush
        sample5 = {
            'timestamp': time.time(),
            'state': {'cpu': 4.0},
            'prediction': {'cpu': 4.0},
            'actual': {'cpu': 4.0},
            'error': {'cpu': 0.1},
            'task_id': 'task_4',
            'reward': 0.5,
            'value_score': 0.5,
        }
        ret5 = mem.store_sample(sample5)
        logger.info("写入第5条（触发flush），返回=%s, 缓冲区大小=%d", ret5, len(mem._sample_buffer))

        buffer_count_after = len(mem._sample_buffer)
        db_count_after = mem.get_sample_count()
        logger.info("触发后：缓冲区=%d, 数据库=%d", buffer_count_after, db_count_after)

        mem.flush()
        mem.close()

        # 预期bug：缓冲区4条数据丢失，只有第5条进了数据库
        # 正确行为：5条全部进数据库
        passed = (db_count_after == 5)
    assert result(
            "Bug1a: 缓冲区数据是否丢失",
            passed,
            f"预期数据库5条，实际{db_count_after}条（缓冲区残留{buffer_count_after}条）"
        )


def test_bug1b_flush_failure_clears_buffer_loses_data():
    """CRITICAL: _flush_all的finally块无条件清空缓冲区，flush失败时数据丢失。

    场景：模拟flush时数据库异常（如磁盘满/数据库锁），
    _flush_all捕获异常后finally仍清空缓冲区，导致数据永久丢失。
    """
    section("Bug1b: flush失败时缓冲区被清空导致数据丢失")
    from src.memory.short_term import ShortTermMemory

    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        mem = ShortTermMemory(db_path=db_path, batch_size=10, flush_interval=9999)

        # 写入3条到缓冲区
        for i in range(3):
            sample = {
                'timestamp': time.time(),
                'state': {'cpu': float(i)},
                'prediction': {'cpu': float(i)},
                'actual': {'cpu': float(i)},
                'error': {'cpu': 0.1},
                'task_id': f'task_{i}',
                'reward': 0.5,
                'value_score': 0.5,
            }
            mem.store_sample(sample)

        buffer_before = len(mem._sample_buffer)
        logger.info("注入故障前：缓冲区=%d条", buffer_before)

        # 注入故障：替换_flush_samples使其抛异常（sqlite3.Connection.commit只读，不能monkey-patch）
        original_flush_samples = mem._flush_samples
        call_count = {'n': 0}

        def faulty_flush_samples():
            call_count['n'] += 1
            if call_count['n'] == 1:
                raise sqlite3.OperationalError("disk I/O error (simulated)")
            return original_flush_samples()

        mem._flush_samples = faulty_flush_samples

        # 尝试flush，应失败
        try:
            mem.flush()
        except Exception as e:
            logger.info("flush抛异常（预期）: %s", type(e).__name__)

        buffer_after = len(mem._sample_buffer)
        logger.info("注入故障后：缓冲区=%d条", buffer_after)

        # 恢复正常_flush_samples
        mem._flush_samples = original_flush_samples

        # 再次flush，看缓冲区数据是否还在
        try:
            mem.flush()
        except Exception as e:
            logger.info("第二次flush异常: %s", e)

        db_count = mem.get_sample_count()
        mem.close()

        # 预期bug：第一次flush失败后缓冲区被清空，3条数据永久丢失
        # 正确行为：第一次flush失败后缓冲区保留，第二次flush成功写入
        passed = (db_count == 3)
    assert result(
            "Bug1b: flush失败后数据是否保留",
            passed,
            f"预期数据库3条，实际{db_count}条（缓冲区残留{buffer_after}条）"
        )


def test_bug1c_handle_flush_error_no_retry():
    """CRITICAL: _handle_flush_error恢复后不重试flush，数据静默丢失。

    场景：flush时遇到'no such table'错误，_handle_flush_error会重建表然后return，
    但不重试flush，缓冲区数据在finally中被清空，数据丢失。
    """
    section("Bug1c: _handle_flush_error恢复后不重试导致数据丢失")
    from src.memory.short_term import ShortTermMemory

    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        mem = ShortTermMemory(db_path=db_path, batch_size=10, flush_interval=9999)

        # 写入2条到缓冲区
        for i in range(2):
            sample = {
                'timestamp': time.time(),
                'state': {'cpu': float(i)},
                'prediction': {'cpu': float(i)},
                'actual': {'cpu': float(i)},
                'error': {'cpu': 0.1},
                'task_id': f'task_{i}',
                'reward': 0.5,
                'value_score': 0.5,
            }
            mem.store_sample(sample)

        # 删除samples表模拟表丢失
        mem._conn.execute("DROP TABLE samples")
        mem._conn.commit()

        buffer_before = len(mem._sample_buffer)
        logger.info("删除表后：缓冲区=%d条", buffer_before)

        # 尝试flush，应触发_handle_flush_error重建表，但不重试
        try:
            mem.flush()
        except Exception as e:
            logger.info("flush异常: %s", type(e).__name__)

        buffer_after = len(mem._sample_buffer)
        db_count = mem.get_sample_count()
        logger.info("flush后：缓冲区=%d条, 数据库=%d条", buffer_after, db_count)

        mem.close()

        # 预期bug：表重建后不重试flush，缓冲区被清空，2条数据丢失
        # 正确行为：表重建后重试flush，2条数据成功写入
        passed = (db_count == 2)
    assert result(
            "Bug1c: 表恢复后数据是否保留",
            passed,
            f"预期数据库2条，实际{db_count}条（缓冲区残留{buffer_after}条）"
        )


# =============================================================================
# Bug 2: agent_loop.py _cleanup_resources无try/finally
# =============================================================================

def test_bug2_cleanup_resources_leak_on_exception():
    """SEVERE: _cleanup_resources无try/finally，异常路径资源全部泄漏。

    场景：stop()调用_cleanup_resources，若memory.flush()抛异常，
    则memory.close()、dashboard.stop()、_remove_log_handler()都不会执行，
    导致数据库连接、dashboard线程、日志handler全部泄漏。
    """
    section("Bug2: _cleanup_resources异常路径资源泄漏")
    from src.core.agent_loop import AgentLoop

    with tempfile.TemporaryDirectory() as tmp:
        config_path = os.path.join(tmp, "config.yaml")
        with open(config_path, 'w') as f:
            f.write("""
agent:
  cycle_interval: 0.01
  max_cycles: 1
memory:
  db_path: %s
  batch_size: 5
  flush_interval: 1.0
task_pool:
  max_size: 10
  max_age_hours: 1
safety:
  max_cpu_percent: 99
  max_memory_percent: 99
  freeze_threshold: 0.5
  exploration_ratio: 0.3
optimization:
  snapshot_dir: %s
  max_snapshots: 5
  mae_degradation_threshold: 0.2
dashboard:
  enabled: false
logging:
  level: INFO
""" % (os.path.join(tmp, "mem.db"), os.path.join(tmp, "snaps")))

        agent = AgentLoop(config_path)

        # 让memory.flush抛异常
        original_flush = agent.memory.flush
        flush_called = {'n': 0}
        close_called = {'n': 0}

        def faulty_flush():
            flush_called['n'] += 1
            raise RuntimeError("simulated flush failure")

        def tracked_close():
            close_called['n'] += 1

        agent.memory.flush = faulty_flush
        agent.memory.close = tracked_close

        # 调用stop()，应触发_cleanup_resources
        try:
            agent.stop()
            exception_raised = False
        except Exception as e:
            exception_raised = True
            logger.info("stop()抛异常: %s: %s", type(e).__name__, e)

        logger.info("flush调用次数=%d, close调用次数=%d", flush_called['n'], close_called['n'])

        # 预期bug：flush抛异常后close未被调用，资源泄漏
        # 正确行为：即使flush抛异常，close也应被调用
        passed = (close_called['n'] >= 1)
    assert result(
            "Bug2: 异常路径下close是否被调用",
            passed,
            f"flush调用{flush_called['n']}次, close调用{close_called['n']}次"
        )


# =============================================================================
# Bug 3: task_pool.py 新任务因score=0被立即驱逐
# =============================================================================

def test_bug3_new_task_evicted_by_score_zero():
    """SEVERE: 新任务因score=0被_enforce_size_limit立即驱逐。

    场景：任务池已满（达到max_size），添加新任务时触发_enforce_size_limit，
    新任务execution_count=0导致score=0，被rank_tasks排在最后，立即被驱逐。
    """
    section("Bug3: 新任务因score=0被立即驱逐")
    from src.task.task_pool import TaskPool
    from src.task.task_definition import Task
    from src.core.constants import TaskType

    with tempfile.TemporaryDirectory() as tmp:
        pool = TaskPool(max_size=3, max_age_hours=1)

        # 先添加3个"老"任务并模拟执行过（有execution_count）
        for i in range(3):
            task = Task(
                id=f"old_task_{i}",
                name=f"old_task_{i}",
                type=TaskType.SAFE,
                action=lambda: None,
            )
            task.execution_count = 10 + i
            task.success_count = 8 + i
            task.avg_reward = 0.5 + i * 0.1
            pool.update([task])

        logger.info("添加3个老任务后，池大小=%d", len(pool.tasks))

        # 添加新任务（execution_count=0，score=0）
        new_task = Task(
            id="new_task",
            name="new_task",
            type=TaskType.SAFE,
            action=lambda: None,
        )
        pool.update([new_task])

        new_task_in_pool = "new_task" in pool.tasks
        logger.info("添加新任务后，池大小=%d, 新任务是否在池中=%s",
                    len(pool.tasks), new_task_in_pool)

        # 预期bug：新任务因score=0被立即驱逐
        # 正确行为：新任务应保留（新任务应有保护期或最低分数）
        passed = new_task_in_pool
    assert result(
            "Bug3: 新任务是否被保留",
            passed,
            f"新任务在池中={new_task_in_pool}, 池大小={len(pool.tasks)}"
        )


# =============================================================================
# Bug 4: snapshot_manager.py 负max_count导致min([])崩溃
# =============================================================================

def test_bug4_negative_max_count_crash():
    """SEVERE: max_count为负数时_enforce_limit崩溃（min([])抛ValueError）。

    场景：配置错误导致max_count=-1，创建快照后_enforce_limit中
    sorted[-1:]返回空列表，min([])抛ValueError。
    """
    section("Bug4: 负max_count导致崩溃")
    from src.optimization.snapshot_manager import SnapshotManager

    with tempfile.TemporaryDirectory() as tmp:
        snap_dir = os.path.join(tmp, "snaps")
        # 使用负max_count
        try:
            mgr = SnapshotManager(snapshot_dir=snap_dir, max_count=-1)
            logger.info("SnapshotManager创建成功，max_count=%d", mgr.max_count)

            # 创建快照触发_enforce_limit
            snap_result = mgr.create_snapshot(
                model_state={'weights': [1, 2, 3]},
                tag="test_negative",
                metrics={'mae': 0.1, 'coverage': 0.9}
            )
            logger.info("快照创建结果: %s", snap_result)
            crashed = False
            crash_msg = ""
        except ValueError as e:
            crashed = True
            crash_msg = str(e)
            logger.info("崩溃（预期bug）: ValueError: %s", e)
        except Exception as e:
            crashed = True
            crash_msg = f"{type(e).__name__}: {e}"
            logger.info("崩溃: %s", crash_msg)

        # 预期bug：min([])抛ValueError崩溃
        # 正确行为：负max_count应被规范化为0或最小值1，不崩溃
        passed = (not crashed)
    assert result(
            "Bug4: 负max_count是否导致崩溃",
            passed,
            f"崩溃={crashed}, 消息={crash_msg}"
        )


# =============================================================================
# Bug 5: auto_rollback.py 无退化检测窗口，单次毛刺即触发回滚
# =============================================================================

def test_bug5_single_spike_triggers_rollback():
    """SEVERE: 无退化检测窗口，单次毛刺即触发回滚。

    场景：baseline_mae=0.1，threshold=0.2。某周期因外部噪声MAE突增到0.4，
    下一周期立即恢复正常0.1。但auto_rollback在第一个毛刺周期就触发回滚，
    造成不必要的回滚操作。
    """
    section("Bug5: 单次毛刺即触发回滚")
    from src.optimization.auto_rollback import AutoRollback
    from src.optimization.snapshot_manager import SnapshotManager

    with tempfile.TemporaryDirectory() as tmp:
        snap_dir = os.path.join(tmp, "snaps")
        snap_mgr = SnapshotManager(snapshot_dir=snap_dir, max_count=5)
        rollback = AutoRollback(
            snapshot_manager=snap_mgr,
            mae_degradation_threshold=0.2
        )

        # 设置baseline
        snap_id = snap_mgr.create_snapshot(
            model_state={'w': 1.0},
            tag="baseline",
            metrics={'mae': 0.1}
        )
        rollback.set_baseline(mae=0.1, snapshot_id=snap_id)
        logger.info("设置baseline: mae=0.1, snap_id=%s", snap_id)

        rollback_count_before = rollback.get_rollback_count()

        # 第1周期：正常
        r1 = rollback.check_and_rollback(current_mae=0.1)
        logger.info("周期1 (mae=0.1): %s", r1)

        # 第2周期：毛刺（突增到0.4，超过threshold=0.2）
        r2 = rollback.check_and_rollback(current_mae=0.4)
        logger.info("周期2 (mae=0.4 毛刺): %s", r2)

        # 第3周期：恢复正常
        r3 = rollback.check_and_rollback(current_mae=0.1)
        logger.info("周期3 (mae=0.1 恢复): %s", r3)

        rollback_count_after = rollback.get_rollback_count()
        logger.info("回滚次数: before=%d, after=%d",
                    rollback_count_before, rollback_count_after)

        # 预期bug：单次毛刺即触发回滚（rollback_count增加）
        # 正确行为：需要连续N个周期退化才触发回滚（窗口机制）
        passed = (rollback_count_after == rollback_count_before)
    assert result(
            "Bug5: 单次毛刺是否避免回滚",
            passed,
            f"回滚次数变化: {rollback_count_before} -> {rollback_count_after}"
        )


# =============================================================================
# Bug 6: micro_optimizer.py level=0.0误判 + 平滑器不学习
# =============================================================================

def test_bug6a_level_zero_misjudged():
    """SEVERE: if smoother.level对level=0.0误判（0.0为falsy）。

    场景：smoother.level=0.0（合法值），但if smoother.level判断为False，
    导致current_error=0.0，优化器误认为无误差，不调整alpha。
    """
    section("Bug6a: level=0.0被误判为falsy")
    from src.optimization.micro_optimizer import MicroOptimizer
    from src.prediction.base_predictor import AdaptiveExponentialSmoother

    opt = MicroOptimizer()

    # 创建一个level=0.0的smoother
    smoother = AdaptiveExponentialSmoother(alpha_init=0.3)
    smoother.level = 0.0  # 合法值
    smoother.last_error = 0.05  # 上次有误差

    alpha_before = smoother.alpha
    logger.info("测试前: level=%s, alpha=%s, last_error=%s",
                smoother.level, smoother.alpha, smoother.last_error)

    # 实际值=1.0，level=0.0，真实误差=1.0，应大幅增加alpha（误差变大）
    opt.optimize_adaptive_smoother(smoother, actual=1.0)

    alpha_after = smoother.alpha
    logger.info("测试后: alpha=%s (变化=%s)", smoother.alpha, alpha_after - alpha_before)

    # 预期bug：level=0.0被误判为falsy，current_error=0.0（应为1.0），
    #          误差变小(0.0<0.05)，alpha减小（应增大）
    # 正确行为：level=0.0是合法值，current_error=1.0，误差变大(1.0>0.05)，alpha应增大
    alpha_increased = alpha_after > alpha_before
    passed = alpha_increased
    assert result(
        "Bug6a: level=0.0时alpha是否正确增大",
        passed,
        f"alpha变化: {alpha_before} -> {alpha_after} (delta={alpha_after - alpha_before}), 预期应增大"
    )


def test_bug6b_smoother_does_not_learn():
    """SEVERE: optimize_adaptive_smoother不更新level与last_error，平滑器实际不学习。

    场景：调用optimize_adaptive_smoother后，只更新alpha，
    但smoother.level和smoother.last_error未更新，
    导致下次调用时仍用旧level计算误差，平滑器永远不学习。
    """
    section("Bug6b: 平滑器不更新level/last_error")
    from src.optimization.micro_optimizer import MicroOptimizer
    from src.prediction.base_predictor import AdaptiveExponentialSmoother

    opt = MicroOptimizer()

    smoother = AdaptiveExponentialSmoother(alpha_init=0.3)
    smoother.level = 0.5
    smoother.last_error = 0.1

    level_before = smoother.level
    last_error_before = smoother.last_error

    logger.info("测试前: level=%s, last_error=%s", level_before, last_error_before)

    # 多次调用优化，实际值持续变化
    for i in range(5):
        actual = 0.5 + i * 0.1  # 0.5, 0.6, 0.7, 0.8, 0.9
        opt.optimize_adaptive_smoother(smoother, actual=actual)
        logger.info("第%d次: actual=%s, level=%s, last_error=%s, alpha=%s",
                    i + 1, actual, smoother.level, smoother.last_error, smoother.alpha)

    level_after = smoother.level
    last_error_after = smoother.last_error

    # 预期bug：level和last_error永远不变
    # 正确行为：level应向actual靠拢，last_error应更新
    level_updated = abs(level_after - level_before) > 1e-6
    last_error_updated = abs(last_error_after - last_error_before) > 1e-6

    passed = level_updated and last_error_updated
    assert result(
        "Bug6b: level/last_error是否更新",
        passed,
        f"level: {level_before}->{level_after}, last_error: {last_error_before}->{last_error_after}"
    )


# =============================================================================
# Bug 7: 综合压力场景 - 多bug同时触发
# =============================================================================

def test_bug7_comprehensive_stress():
    """综合场景：模拟真实运行，多个bug同时暴露。

    场景：运行30个周期，每周期写入sample+创建任务+创建快照，
    中途注入数据库故障，观察数据丢失情况。
    """
    section("Bug7: 综合压力场景（30周期+故障注入）")
    from src.memory.short_term import ShortTermMemory
    from src.task.task_pool import TaskPool
    from src.task.task_definition import Task
    from src.core.constants import TaskType
    from src.optimization.snapshot_manager import SnapshotManager

    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "stress.db")
        snap_dir = os.path.join(tmp, "snaps")

        mem = ShortTermMemory(db_path=db_path, batch_size=10, flush_interval=9999)
        pool = TaskPool(max_size=5, max_age_hours=1)
        snap_mgr = SnapshotManager(snapshot_dir=snap_dir, max_count=3)

        samples_written = 0
        tasks_added = 0
        snapshots_created = 0

        # 中途注入故障
        inject_failure_at = 15
        failure_injected = False

        for cycle in range(30):
            # 写入sample
            sample = {
                'timestamp': time.time(),
                'state': {'cpu': float(cycle)},
                'prediction': {'cpu': float(cycle)},
                'actual': {'cpu': float(cycle)},
                'error': {'cpu': 0.1},
                'task_id': f'task_{cycle}',
                'reward': 0.5,
                'value_score': 0.5,
            }
            mem.store_sample(sample)
            samples_written += 1

            # 添加任务
            task = Task(
                id=f"task_{cycle}",
                name=f"task_{cycle}",
                type=TaskType.SAFE,
                action=lambda: None,
            )
            pool.update([task])
            tasks_added += 1

            # 创建快照
            try:
                snap_mgr.create_snapshot(
                    model_state={'cycle': cycle},
                    tag=f"cycle_{cycle}",
                    metrics={'mae': 0.1 + cycle * 0.01}
                )
                snapshots_created += 1
            except Exception as e:
                logger.info("周期%d: 快照创建失败: %s", cycle, e)

            # 中途注入数据库故障（替换_flush_samples使其抛异常）
            if cycle == inject_failure_at and not failure_injected:
                logger.info(">>> 注入数据库故障 (cycle=%d) <<<", cycle)
                original_flush_samples = mem._flush_samples
                fail_count = {'n': 0}

                def faulty_flush_samples():
                    fail_count['n'] += 1
                    if fail_count['n'] <= 2:
                        raise sqlite3.OperationalError("simulated disk error")
                    return original_flush_samples()

                mem._flush_samples = faulty_flush_samples
                failure_injected = True

        # 最终flush
        try:
            mem.flush()
        except Exception as e:
            logger.info("最终flush异常: %s", e)

        db_samples = mem.get_sample_count()
        pool_size = len(pool.tasks)
        snap_count = len(snap_mgr._snapshots)

        mem.close()

        logger.info("综合场景结果:")
        logger.info("  写入samples: %d, 数据库实际: %d, 丢失: %d",
                    samples_written, db_samples, samples_written - db_samples)
        logger.info("  添加tasks: %d, 池中剩余: %d, 驱逐: %d",
                    tasks_added, pool_size, tasks_added - pool_size)
        logger.info("  创建snapshots: %d, 保留: %d",
                    snapshots_created, snap_count)

        # 关键验收：数据丢失率
        sample_loss_rate = (samples_written - db_samples) / samples_written
        task_eviction_rate = (tasks_added - pool_size) / tasks_added

        logger.info("  sample丢失率: %.1f%%", sample_loss_rate * 100)
        logger.info("  task驱逐率: %.1f%%", task_eviction_rate * 100)

        # 预期：由于bug，sample丢失率应较高
        # 验收标准：sample丢失率应<10%（修复后）
        passed = (sample_loss_rate < 0.1)
    assert result(
            "Bug7: 综合场景sample丢失率",
            passed,
            f"sample丢失率={sample_loss_rate * 100:.1f}%, task驱逐率={task_eviction_rate * 100:.1f}%"
        )


# =============================================================================
# Bug 8: 并发竞态 - 多线程同时store_sample
# =============================================================================

def test_bug8_concurrent_store_sample_race():
    """并发场景：多线程同时调用store_sample，验证锁与缓冲区一致性。

    场景：4个线程各写入50条sample，共200条。
    验证数据库最终sample数是否=200，缓冲区是否为空。
    """
    section("Bug8: 并发store_sample竞态")
    from src.memory.short_term import ShortTermMemory

    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "race.db")
        mem = ShortTermMemory(db_path=db_path, batch_size=20, flush_interval=9999)

        threads = []
        errors = []

        def writer(thread_id):
            try:
                for i in range(50):
                    sample = {
                        'timestamp': time.time(),
                        'state': {'cpu': float(thread_id * 100 + i)},
                        'prediction': {'cpu': float(thread_id * 100 + i)},
                        'actual': {'cpu': float(thread_id * 100 + i)},
                        'error': {'cpu': 0.1},
                        'task_id': f't{thread_id}_task_{i}',
                        'reward': 0.5,
                        'value_score': 0.5,
                    }
                    mem.store_sample(sample)
            except Exception as e:
                errors.append((thread_id, str(e)))

        for tid in range(4):
            t = threading.Thread(target=writer, args=(tid,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # 最终flush
        try:
            mem.flush()
        except Exception as e:
            logger.info("最终flush异常: %s", e)

        db_count = mem.get_sample_count()
        buffer_count = len(mem._sample_buffer)
        mem.close()

        logger.info("并发写入完成: 数据库=%d, 缓冲区残留=%d, 错误数=%d",
                    db_count, buffer_count, len(errors))

        # 预期：由于bug1a（_should_flush触发直接插入），部分数据可能丢失
        # 验收：数据库应有200条
        passed = (db_count == 200 and buffer_count == 0 and len(errors) == 0)
    assert result(
            "Bug8: 并发写入数据完整性",
            passed,
            f"数据库={db_count}/200, 缓冲区残留={buffer_count}, 错误={len(errors)}"
        )


# =============================================================================
# 主函数
# =============================================================================

def main():
    logger.info("=" * 70)
    logger.info("阶段114：超大规模复杂场景压力测试 - 开始")
    logger.info("目的：验证阶段113静态核验发现的严重bug在真实运行中触发")
    logger.info("=" * 70)

    tests = [
        ("Bug1a", test_bug1a_should_flush_triggers_direct_insert_loses_buffer),
        ("Bug1b", test_bug1b_flush_failure_clears_buffer_loses_data),
        ("Bug1c", test_bug1c_handle_flush_error_no_retry),
        ("Bug2", test_bug2_cleanup_resources_leak_on_exception),
        ("Bug3", test_bug3_new_task_evicted_by_score_zero),
        ("Bug4", test_bug4_negative_max_count_crash),
        ("Bug5", test_bug5_single_spike_triggers_rollback),
        ("Bug6a", test_bug6a_level_zero_misjudged),
        ("Bug6b", test_bug6b_smoother_does_not_learn),
        ("Bug7", test_bug7_comprehensive_stress),
        ("Bug8", test_bug8_concurrent_store_sample_race),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            logger.exception("测试%s抛出未捕获异常", name)
            results[name] = False

    # 汇总
    logger.info("")
    logger.info("=" * 70)
    logger.info("阶段114测试汇总")
    logger.info("=" * 70)
    passed_count = sum(1 for v in results.values() if v)
    failed_count = len(results) - passed_count
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL (bug已复现)"
        logger.info("  %s: %s", name, status)
    logger.info("-" * 70)
    logger.info("通过: %d, 失败(bug复现): %d, 总计: %d",
                passed_count, failed_count, len(results))
    logger.info("=" * 70)

    # 失败的测试即为bug已复现的测试，这些是需要修复的
    if failed_count > 0:
        logger.info("")
        logger.info("已复现的bug（需在阶段115修复）:")
        for name, passed in results.items():
            if not passed:
                logger.info("  - %s", name)

    return results


if __name__ == '__main__':
    main()
