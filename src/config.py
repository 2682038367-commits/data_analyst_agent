"""全局配置：从环境变量读取 LLM 与数据库配置。"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# ---- LLM ----
# 优先使用 DeepSeek Key，其次 OpenAI Key
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
# Evaluation 可使用独立 Judge 模型，避免被测模型自评偏差。
EVAL_JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", LLM_MODEL)

# 代理：默认留空 = 不使用环境代理（DeepSeek 国内直连，且 socks 协议不受支持）。
# 如果将来需要走代理（例如访问 OpenAI），在这里显式指定 http 代理：
#   LLM_PROXY=http://127.0.0.1:7890
LLM_PROXY = os.getenv("LLM_PROXY", "")

# ---- Database ----
DB_PATH = Path(os.getenv("DB_PATH", BASE_DIR / "data" / "ecommerce.db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---- Agent ----
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
# 查询结果合理性检查的行数上限；执行阶段只读取上限 + 1 行用于判定超限。
RESULT_MAX_ROWS = max(1, int(os.getenv("RESULT_MAX_ROWS", "10000")))
# 核心查询之后最多自动追加的归因/下钻查询轮数。
MAX_ANALYSIS_DEPTH = max(0, int(os.getenv("MAX_ANALYSIS_DEPTH", "3")))
