from enum import Enum


class TaskType(Enum):
    SAFE = "safe"
    GAIN = "gain"
    EFF = "eff"
    EXPLORE = "explore"


class RunMode(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    EMERGENCY = "emergency"


class SensorType(Enum):
    SYSTEM = "system"
    LOG = "log"
    FILE = "file"
    SELF = "self"


DEFAULT_CONFIG_PATH = "config/config.prod.yaml"

SAFE_GAP_THRESHOLD = 0.3
GAIN_GAP_THRESHOLD = 0.3
EFF_GAP_THRESHOLD = 0.3

CPU_HIGH_RISK = 90.0
CPU_MED_RISK = 70.0
MEM_HIGH_RISK = 90.0
MEM_MED_RISK = 80.0
DISK_HIGH_RISK = 95.0
ERROR_HIGH_RATE = 10
ERROR_MED_RATE = 5

TARGET_SELF_CPU = 5.0
TARGET_SELF_MEM_MB = 256.0

EXPLORATION_BASE_RATIO = 0.3
EXPLORATION_STABLE_THRESHOLD = 0.05

MAX_CONSECUTIVE_FAILURES = 5
MAX_RESTRUCTURE_ITERATIONS = 3

VALUE_RETENTION_RATIO = 0.3
MAX_SHORT_TERM_SAMPLES = 10000

SNAPSHOT_TAG_AUTO = "auto"
SNAPSHOT_TAG_MANUAL = "manual"
SNAPSHOT_TAG_RESTRUCTURE = "restructure"

DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8080

DAEMON_PID_FILE = "/tmp/self_evolving_agent.pid"
DAEMON_WORKING_DIR = "/"
DAEMON_UMASK = 0o022

DEFAULT_LOG_PATHS = ["/var/log/syslog", "/var/log/messages"]
DEFAULT_WATCH_PATHS = ["/tmp", "/var/tmp"]

DEFAULT_MEMORY_DB_PATH = "data/memory.db"
