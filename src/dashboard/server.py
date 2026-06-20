#!/usr/bin/env python3
"""性能监控仪表盘 - 轻量化HTTP服务"""

import abc
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Dict, Optional

from src.core.constants import DASHBOARD_HOST, DASHBOARD_PORT

logger = logging.getLogger(__name__)

_DASHBOARD_HTML = (
    '<!DOCTYPE html>\n'
    '<html lang="zh-CN">\n'
    '<head>\n'
    '    <meta charset="UTF-8">\n'
    '    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
    '    <title>自驱Agent监控仪表盘</title>\n'
    '    <style>\n'
    '        * { margin: 0; padding: 0; box-sizing: border-box; }\n'
    '        body {\n'
    '            font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, sans-serif;\n'
    '            background: #0f172a;\n'
    '            color: #e2e8f0;\n'
    '            padding: 20px;\n'
    '        }\n'
    '        .header {\n'
    '            text-align: center;\n'
    '            margin-bottom: 30px;\n'
    '        }\n'
    '        .header h1 {\n'
    '            font-size: 24px;\n'
    '            color: #60a5fa;\n'
    '            margin-bottom: 8px;\n'
    '        }\n'
    '        .header .status {\n'
    '            font-size: 14px;\n'
    '            color: #94a3b8;\n'
    '        }\n'
    '        .grid {\n'
    '            display: grid;\n'
    '            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));\n'
    '            gap: 16px;\n'
    '            max-width: 1200px;\n'
    '            margin: 0 auto;\n'
    '        }\n'
    '        .card {\n'
    '            background: #1e293b;\n'
    '            border-radius: 12px;\n'
    '            padding: 20px;\n'
    '            border: 1px solid #334155;\n'
    '        }\n'
    '        .card h3 {\n'
    '            font-size: 13px;\n'
    '            color: #94a3b8;\n'
    '            text-transform: uppercase;\n'
    '            letter-spacing: 0.5px;\n'
    '            margin-bottom: 12px;\n'
    '        }\n'
    '        .metric {\n'
    '            font-size: 32px;\n'
    '            font-weight: 700;\n'
    '            color: #f8fafc;\n'
    '        }\n'
    '        .metric.small {\n'
    '            font-size: 20px;\n'
    '        }\n'
    '        .metric.good { color: #4ade80; }\n'
    '        .metric.warn { color: #fbbf24; }\n'
    '        .metric.bad { color: #f87171; }\n'
    '        .detail {\n'
    '            margin-top: 8px;\n'
    '            font-size: 12px;\n'
    '            color: #64748b;\n'
    '        }\n'
    '        .chart-container {\n'
    '            grid-column: 1 / -1;\n'
    '            height: 300px;\n'
    '        }\n'
    '        canvas {\n'
    '            width: 100%;\n'
    '            height: 100%;\n'
    '        }\n'
    '        .state-badge {\n'
    '            display: inline-block;\n'
    '            padding: 4px 12px;\n'
    '            border-radius: 20px;\n'
    '            font-size: 12px;\n'
    '            font-weight: 600;\n'
    '        }\n'
    '        .state-idle { background: #3b82f6; }\n'
    '        .state-active { background: #22c55e; }\n'
    '        .state-emergency { background: #ef4444; }\n'
    '        .state-review { background: #f59e0b; }\n'
    '        .update-time {\n'
    '            text-align: center;\n'
    '            margin-top: 20px;\n'
    '            font-size: 12px;\n'
    '            color: #475569;\n'
    '        }\n'
    '    </style>\n'
    '</head>\n'
    '<body>\n'
    '    <div class="header">\n'
    '        <h1>自驱永续进化Agent</h1>\n'
    '        <div class="status">\n'
    '            状态: <span id="state-badge" class="state-badge state-idle">IDLE</span>\n'
    '            &nbsp;|&nbsp; 运行周期: <span id="total-cycles">0</span>\n'
    '        </div>\n'
    '    </div>\n'
    '\n'
    '    <div class="grid">\n'
    '        <div class="card">\n'
    '            <h3>预测误差 (MAE)</h3>\n'
    '            <div id="mae-value" class="metric">--</div>\n'
    '            <div class="detail">越低越好</div>\n'
    '        </div>\n'
    '        <div class="card">\n'
    '            <h3>任务成功率</h3>\n'
    '            <div id="success-value" class="metric">--</div>\n'
    '            <div class="detail">近期执行成功率</div>\n'
    '        </div>\n'
    '        <div class="card">\n'
    '            <h3>综合奖励</h3>\n'
    '            <div id="reward-value" class="metric">--</div>\n'
    '            <div class="detail">驱力综合评分</div>\n'
    '        </div>\n'
    '        <div class="card">\n'
    '            <h3>记忆样本</h3>\n'
    '            <div id="memory-value" class="metric small">--</div>\n'
    '            <div class="detail">短期记忆存储量</div>\n'
    '        </div>\n'
    '        <div class="card">\n'
    '            <h3>任务池</h3>\n'
    '            <div id="tasks-value" class="metric small">--</div>\n'
    '            <div class="detail">当前活跃任务数</div>\n'
    '        </div>\n'
    '        <div class="card">\n'
    '            <h3>预测指标</h3>\n'
    '            <div id="predictions-value" class="metric small">--</div>\n'
    '            <div class="detail">底层预测输出数</div>\n'
    '        </div>\n'
    '        <div class="card chart-container">\n'
    '            <h3>历史趋势</h3>\n'
    '            <canvas id="history-chart"></canvas>\n'
    '        </div>\n'
    '    </div>\n'
    '\n'
    '    <div class="update-time">最后更新: <span id="last-update">--</span></div>\n'
    '\n'
    '    <script>\n'
    '        const ctx = document.getElementById(\'history-chart\').getContext(\'2d\');\n'
    '        let historyData = [];\n'
    '\n'
    '        function drawChart() {\n'
    '            const canvas = document.getElementById(\'history-chart\');\n'
    '            const w = canvas.width = canvas.offsetWidth;\n'
    '            const h = canvas.height = canvas.offsetHeight;\n'
    '\n'
    '            ctx.clearRect(0, 0, w, h);\n'
    '            if (historyData.length < 2) return;\n'
    '\n'
    '            const padding = 40;\n'
    '            const chartW = w - padding * 2;\n'
    '            const chartH = h - padding * 2;\n'
    '\n'
    '            const maxReward = Math.max(...historyData.map(d => d.reward), 1);\n'
    '            const maxMae = Math.max(...historyData.map(d => d.mae), 1);\n'
    '\n'
    '            ctx.strokeStyle = \'#334155\';\n'
    '            ctx.lineWidth = 1;\n'
    '            for (let i = 0; i <= 4; i++) {\n'
    '                const y = padding + (chartH / 4) * i;\n'
    '                ctx.beginPath();\n'
    '                ctx.moveTo(padding, y);\n'
    '                ctx.lineTo(w - padding, y);\n'
    '                ctx.stroke();\n'
    '            }\n'
    '\n'
    '            ctx.strokeStyle = \'#f87171\';\n'
    '            ctx.lineWidth = 2;\n'
    '            ctx.beginPath();\n'
    '            historyData.forEach((d, i) => {\n'
    '                const x = padding + (chartW / (historyData.length - 1)) * i;\n'
    '                const y = padding + chartH - (d.mae / maxMae) * chartH;\n'
    '                if (i === 0) ctx.moveTo(x, y);\n'
    '                else ctx.lineTo(x, y);\n'
    '            });\n'
    '            ctx.stroke();\n'
    '\n'
    '            ctx.strokeStyle = \'#4ade80\';\n'
    '            ctx.beginPath();\n'
    '            historyData.forEach((d, i) => {\n'
    '                const x = padding + (chartW / (historyData.length - 1)) * i;\n'
    '                const y = padding + chartH - (d.reward / maxReward) * chartH;\n'
    '                if (i === 0) ctx.moveTo(x, y);\n'
    '                else ctx.lineTo(x, y);\n'
    '            });\n'
    '            ctx.stroke();\n'
    '\n'
    '            ctx.fillStyle = \'#f87171\';\n'
    '            ctx.fillRect(w - 120, 10, 12, 12);\n'
    '            ctx.fillStyle = \'#94a3b8\';\n'
    '            ctx.font = \'12px sans-serif\';\n'
    '            ctx.fillText(\'MAE\', w - 100, 21);\n'
    '\n'
    '            ctx.fillStyle = \'#4ade80\';\n'
    '            ctx.fillRect(w - 60, 10, 12, 12);\n'
    '            ctx.fillStyle = \'#94a3b8\';\n'
    '            ctx.fillText(\'Reward\', w - 40, 21);\n'
    '        }\n'
    '\n'
    '        function updateDashboard() {\n'
    '            fetch(\'/api/data\')\n'
    '                .then(r => r.json())\n'
    '                .then(data => {\n'
    '                    document.getElementById(\'total-cycles\').textContent = data.cycles;\n'
    '                    document.getElementById(\'mae-value\').textContent = data.recent_mae.toFixed(4);\n'
    '                    document.getElementById(\'success-value\').textContent = (data.success_rate * 100).toFixed(1) + \'%\';\n'
    '                    document.getElementById(\'reward-value\').textContent = data.reward.toFixed(4);\n'
    '                    document.getElementById(\'memory-value\').textContent = data.memory_samples;\n'
    '\n'
    '                    const tasks = data.task_pool.total_tasks || 0;\n'
    '                    document.getElementById(\'tasks-value\').textContent = tasks;\n'
    '\n'
    '                    const predCount = Object.keys(data.predictions.low || {}).length;\n'
    '                    document.getElementById(\'predictions-value\').textContent = predCount;\n'
    '\n'
    '                    const badge = document.getElementById(\'state-badge\');\n'
    '                    badge.textContent = data.current_state;\n'
    '                    badge.className = \'state-badge state-\' + data.current_state.toLowerCase();\n'
    '\n'
    '                    const maeEl = document.getElementById(\'mae-value\');\n'
    '                    if (data.recent_mae < 0.05) maeEl.className = \'metric good\';\n'
    '                    else if (data.recent_mae < 0.15) maeEl.className = \'metric warn\';\n'
    '                    else maeEl.className = \'metric bad\';\n'
    '\n'
    '                    const succEl = document.getElementById(\'success-value\');\n'
    '                    if (data.success_rate > 0.7) succEl.className = \'metric good\';\n'
    '                    else if (data.success_rate > 0.4) succEl.className = \'metric warn\';\n'
    '                    else succEl.className = \'metric bad\';\n'
    '\n'
    '                    historyData = data.history || [];\n'
    '                    drawChart();\n'
    '\n'
    '                    document.getElementById(\'last-update\').textContent = new Date().toLocaleTimeString();\n'
    '                })\n'
    '                .catch(err => {\n'
    '                    console.error(\'Fetch error:\', err);\n'
    '                });\n'
    '        }\n'
    '\n'
    '        updateDashboard();\n'
    '        setInterval(updateDashboard, 2000);\n'
    '        window.addEventListener(\'resize\', drawChart);\n'
    '    </script>\n'
    '</body>\n'
    '</html>'
)


class _DashboardDataStore(abc.ABC):
    """仪表盘数据存储"""

    def __init__(self):
        """Initialize the dashboard data store.

        :param: 无
        :type: 无
        :return: None
        :rtype: None
        """
        self._data = self._create_default_data()
        self._lock = threading.Lock()
        self._max_history = 100

    def _create_default_data(self) -> Dict[str, Any]:
        return self._build_default_data()

    def _build_default_data(self) -> Dict[str, Any]:
        data = self._init_base_metrics()
        data['task_pool'] = {}
        data['memory_samples'] = 0
        data['current_state'] = 'IDLE'
        self._init_prediction_data(data)
        return data

    def _init_prediction_data(self, data: Dict[str, Any]) -> None:
        data['predictions'] = {}
        data['errors'] = {}
        data['reward'] = 0.0
        data['history'] = []

    def _init_base_metrics(self) -> Dict[str, Any]:
        return {
            'cycles': 0,
            'recent_mae': 0.0,
            'success_rate': 0.0,
        }

    @abc.abstractmethod
    def update(self, cycle_result: Dict[str, Any]) -> None:
        """更新数据

        :param cycle_result: Result dictionary from a cycle.
        :type cycle_result: Dict[str, Any]
        :return: None
        :rtype: None
        """

    def _do_update(self, cycle_result: Dict[str, Any]) -> None:
        with self._lock:
            self._apply_cycle_data(cycle_result)
            self._append_history(cycle_result)

    def _apply_cycle_data(self, cycle_result: Dict[str, Any]) -> None:
        self._data['cycles'] = cycle_result.get('cycle', 0)
        self._apply_safety_state(cycle_result.get('safety', {}))
        self._apply_prediction_data(cycle_result)

    def _apply_safety_state(self, safety: Dict[str, Any]) -> None:
        if safety.get('emergency_mode', False):
            self._data['current_state'] = 'EMERGENCY'
        else:
            self._data['current_state'] = 'ACTIVE'

    def _apply_prediction_data(self, cycle_result: Dict[str, Any]) -> None:
        self._data['predictions'] = cycle_result.get('predictions', {})
        self._data['errors'] = cycle_result.get('errors', {})
        self._data['reward'] = cycle_result.get('reward', {}).get('total_reward', 0.0)

    def _append_history(self, cycle_result: Dict[str, Any]) -> None:
        entry = self._build_history_entry(cycle_result)
        self._data['history'].append(entry)
        self._trim_history()

    def _build_history_entry(self, cycle_result: Dict[str, Any]) -> Dict[str, Any]:
        errors = cycle_result.get('errors', {})
        mae = sum(abs(v) for v in errors.values()) / max(len(errors), 1) if errors else 0.0
        return {'cycle': cycle_result.get('cycle', 0), 'mae': mae, 'reward': cycle_result.get('reward', {}).get('total_reward', 0.0), 'duration': cycle_result.get('duration_seconds', 0.0)}

    def _trim_history(self) -> None:
        if len(self._data['history']) > self._max_history:
            self._data['history'] = self._data['history'][-self._max_history:]

    def _safe_json_dumps(self, data: Dict[str, Any]) -> bytes:
        """安全的JSON序列化，处理不可序列化对象"""
        def default_encoder(obj):
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)
        return json.dumps(data, default=default_encoder).encode('utf-8')

    @abc.abstractmethod
    def update_stats(self, stats: Dict[str, Any]) -> None:
        """更新统计信息

        :param stats: Statistics dictionary.
        :type stats: Dict[str, Any]
        :return: None
        :rtype: None
        """

    def _do_update_stats(self, stats: Dict[str, Any]) -> None:
        with self._lock:
            self._data['recent_mae'] = stats.get('recent_mae', 0.0)
            self._data['success_rate'] = stats.get('recent_success_rate', 0.0)
            self._data['task_pool'] = stats.get('task_pool', {})
            self._data['memory_samples'] = stats.get('memory_samples', 0)

    def get_data(self) -> Dict[str, Any]:
        """获取当前数据副本

        :param: 无
        :type: 无
        :return: Current data copy.
        :rtype: Dict[str, Any]
        """
        with self._lock:
            return dict(self._data)


class DashboardDataStore(_DashboardDataStore):
    """DashboardDataStore 实现"""

    def update(self, cycle_result: Dict[str, Any]) -> None:
        """更新数据

        :param cycle_result: Result dictionary from a cycle.
        :type cycle_result: Dict[str, Any]
        :return: None
        :rtype: None
        """
        self._do_update(cycle_result)

    def update_stats(self, stats: Dict[str, Any]) -> None:
        """更新统计信息

        :param stats: Statistics dictionary.
        :type stats: Dict[str, Any]
        :return: None
        :rtype: None
        """
        self._do_update_stats(stats)


class _DashboardHandler(BaseHTTPRequestHandler, abc.ABC):
    """HTTP请求处理器"""

    data_store: Optional[_DashboardDataStore] = None
    alert_manager: Optional[Any] = None
    metrics_provider: Optional[Callable[[], Dict[str, Any]]] = None
    skills_provider: Optional[Callable[[], Dict[str, Any]]] = None

    @abc.abstractmethod
    def do_GET(self):
        """处理 GET 请求

        :param: 无
        :type: 无
        :return: None
        :rtype: None
        """

    def _do_get_impl(self):
        route = self._get_route()
        route()

    def _get_route(self) -> Callable:
        if self.path == '/':
            return self._serve_html
        elif self.path == '/api/data':
            return self._serve_json
        elif self.path == '/api/health':
            return self._serve_health
        elif self.path == '/api/alerts':
            return self._serve_alerts
        elif self.path == '/api/metrics':
            return self._serve_metrics
        elif self.path == '/api/skills':
            return self._serve_skills
        return self._send_404

    def _send_404(self) -> None:
        self.send_error(404)

    def _serve_html(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(self._dashboard_html().encode('utf-8'))

    def _dashboard_html(self) -> str:
        """Return the dashboard HTML template."""
        return ''

    def _serve_json(self):
        if self.data_store is None:
            self.send_error(503)
            return
        self._send_json_headers()
        self._write_json_data()

    def _send_json_headers(self) -> None:
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def _write_json_data(self) -> None:
        data = self.data_store.get_data()
        self.wfile.write(self.data_store._safe_json_dumps(data))

    def _serve_health(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'status': 'ok'}).encode('utf-8'))

    def _serve_alerts(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        if self.alert_manager is not None:
            data = self.alert_manager.get_active_alerts()
            stats = self.alert_manager.get_stats()
        else:
            data = []
            stats = {}
        payload = {'alerts': data, 'stats': stats}
        self.wfile.write(json.dumps(payload, default=str).encode('utf-8'))

    def _serve_metrics(self):
        """输出 PerformanceTracker 性能指标。"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        metrics = self.metrics_provider() if self.metrics_provider is not None else {}
        self.wfile.write(json.dumps(metrics, default=str).encode('utf-8'))

    def _serve_skills(self):
        """输出技能注册表统计。"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        skills = self.skills_provider() if self.skills_provider is not None else {}
        self.wfile.write(json.dumps(skills, default=str).encode('utf-8'))


class DashboardHandler(_DashboardHandler):
    """DashboardHandler 实现"""

    def do_GET(self):
        """处理 GET 请求

        :param: 无
        :type: 无
        :return: None
        :rtype: None
        """
        self._do_get_impl()

    def _dashboard_html(self) -> str:
        """Return the dashboard HTML template."""
        return self._get_html_template()

    def _get_html_template(self) -> str:
        """Return the static HTML template string."""
        return self._build_html()

    def _build_html(self) -> str:
        """Build and return the dashboard HTML content."""
        return self._html_content()

    def _html_content(self) -> str:
        """Return the static HTML content string."""
        return _DASHBOARD_HTML


class _DashboardServer(abc.ABC):
    """仪表盘服务器"""

    def __init__(self, host: str = DASHBOARD_HOST, port: int = DASHBOARD_PORT):
        """Initialize the dashboard server.

        :param host: Server host address.
        :type host: str
        :param port: Server port number.
        :type port: int
        :return: None
        :rtype: None
        """
        self.host = host
        self.port = port
        self._init_data_store()
        self._init_server_state()

    def _init_data_store(self) -> None:
        self.data_store = DashboardDataStore()

    def _init_server_state(self) -> None:
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    @abc.abstractmethod
    def start(self) -> None:
        """启动仪表盘服务器

        :param: 无
        :type: 无
        :return: None
        :rtype: None
        """

    def _do_start(self) -> None:
        DashboardHandler.data_store = self.data_store
        self._server = HTTPServer((self.host, self.port), DashboardHandler)
        self._running = True
        self._start_server_thread()
        self._log_server_start()

    def set_alert_manager(self, alert_manager: Any) -> None:
        """Set the alert manager for the dashboard handler.

        :param alert_manager: Alert manager instance.
        :type alert_manager: Any
        :return: None
        :rtype: None
        """
        DashboardHandler.alert_manager = alert_manager

    def set_metrics_provider(self, provider: Callable[[], Dict[str, Any]]) -> None:
        """设置性能指标提供者，用于 /api/metrics 路由。

        :param provider: 返回性能指标字典的可调用对象
        :type provider: Callable[[], Dict[str, Any]]
        :return: None
        :rtype: None
        """
        DashboardHandler.metrics_provider = provider

    def set_skills_provider(self, provider: Callable[[], Dict[str, Any]]) -> None:
        """设置技能统计提供者，用于 /api/skills 路由。

        :param provider: 返回技能统计字典的可调用对象
        :type provider: Callable[[], Dict[str, Any]]
        :return: None
        :rtype: None
        """
        DashboardHandler.skills_provider = provider

    def _start_server_thread(self) -> None:
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def _log_server_start(self) -> None:
        logger.info("Dashboard server started at http://%s:%s", self.host, self.port)

    @abc.abstractmethod
    def stop(self) -> None:
        """停止仪表盘服务器

        :param: 无
        :type: 无
        :return: None
        :rtype: None
        """

    def _do_stop(self) -> None:
        self._shutdown_server()
        self._running = False
        self._join_server_thread()
        self._log_server_stop()

    def _join_server_thread(self) -> None:
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _shutdown_server(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()

    def _log_server_stop(self) -> None:
        logger.info("Dashboard server stopped")

    def update(self, cycle_result: Dict[str, Any]) -> None:
        """更新仪表盘数据

        :param cycle_result: Result dictionary from a cycle.
        :type cycle_result: Dict[str, Any]
        :return: None
        :rtype: None
        """
        self.data_store.update(cycle_result)

    def update_stats(self, stats: Dict[str, Any]) -> None:
        """更新统计信息

        :param stats: Statistics dictionary.
        :type stats: Dict[str, Any]
        :return: None
        :rtype: None
        """
        self.data_store.update_stats(stats)

    def is_running(self) -> bool:
        """Return whether the server is running.

        :param: 无
        :type: 无
        :return: Whether the server is running.
        :rtype: bool
        """
        return self._running


class DashboardServer(_DashboardServer):
    """DashboardServer 实现"""

    def start(self) -> None:
        """启动仪表盘服务器

        :param: 无
        :type: 无
        :return: None
        :rtype: None
        """
        self._do_start()

    def stop(self) -> None:
        """停止仪表盘服务器

        :param: 无
        :type: 无
        :return: None
        :rtype: None
        """
        self._do_stop()
