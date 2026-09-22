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

# 代理：默认留空 = 不使用环境代理（DeepSeek 国内直连，且 socks 协议不受支持）。
# 如果将来需要走代理（例如访问 OpenAI），在这里显式指定 http 代理：
#   LLM_PROXY=http://127.0.0.1:7890
LLM_PROXY = os.getenv("LLM_PROXY", "")

# ---- Database ----
DB_PATH = Path(os.getenv("DB_PATH", BASE_DIR / "data" / "ecommerce.db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---- Agent ----
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
