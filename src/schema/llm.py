"""DeepSeek LLM 工厂（LangChain），读取 uvgraph 根目录 .env。"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from configs import config

load_dotenv(config.ROOT_DIR.parent / ".env")


@lru_cache
def get_deepseek_llm() -> ChatOpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，请在 uvgraph 根目录 .env 中设置")
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        temperature=0.2,
    )
