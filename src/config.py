"""全局配置：从环境变量读取 LLM 与数据库配置。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# ---- LLM ----
def resolve_llm_settings(environ: Mapping[str, str]) -> tuple[str, str, str, str, str]:
    """按供应商选择密钥、端点和模型；GLM 不继承 DeepSeek 的通用覆盖项。"""
    provider = (environ.get("LLM_PROVIDER") or "glm").strip().lower()
    if provider == "glm":
        return (
            provider,
            environ.get("GLM_API_KEY") or environ.get("ZAI_API_KEY") or "",
            environ.get("GLM_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4/",
            environ.get("GLM_MODEL") or "glm-5.3",
            "GLM_API_KEY（或 ZAI_API_KEY）",
        )
    if provider == "deepseek":
        return (
            provider,
            environ.get("DEEPSEEK_API_KEY") or environ.get("OPENAI_API_KEY") or "",
            environ.get("LLM_BASE_URL") or "https://api.deepseek.com",
            environ.get("LLM_MODEL") or "deepseek-chat",
            "DEEPSEEK_API_KEY（兼容接口也可用 OPENAI_API_KEY）",
        )
    raise ValueError("LLM_PROVIDER 只支持 deepseek 或 glm")


def resolve_fallback_settings(
    environ: Mapping[str, str],
) -> tuple[str, str, str]:
    """仅在 GLM 为主模型时使用独立 DeepSeek Key 与端点。"""
    provider = (environ.get("LLM_PROVIDER") or "glm").strip().lower()
    if provider != "glm":
        return "", "", ""
    return (
        environ.get("DEEPSEEK_API_KEY") or "",
        environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
        environ.get("DEEPSEEK_MODEL") or "deepseek-chat",
    )


LLM_PROVIDER, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_KEY_HINT = (
    resolve_llm_settings(os.environ)
)
LLM_FALLBACK_API_KEY, LLM_FALLBACK_BASE_URL, LLM_FALLBACK_MODEL = (
    resolve_fallback_settings(os.environ)
)
LLM_FALLBACK_PROVIDER = "deepseek" if LLM_PROVIDER == "glm" else ""
LLM_AVAILABLE = bool(LLM_API_KEY or LLM_FALLBACK_API_KEY)

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
# Evaluation 可使用独立 Judge 模型，避免被测模型自评偏差。
EVAL_JUDGE_MODEL = os.getenv(
    "EVAL_JUDGE_MODEL", LLM_MODEL if LLM_API_KEY else (LLM_FALLBACK_MODEL or LLM_MODEL)
)

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
MAX_ANALYSIS_DEPTH = min(5, max(0, int(os.getenv("MAX_ANALYSIS_DEPTH", "3"))))
