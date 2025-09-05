import os
import json
from typing import Dict


DEFAULTS: Dict[str, str] = {
    "base_url": "http://10.4.143.24/qwen/v1",
    "api_key": "api_key_123",
    "model": "qwen2.5-vl",
    "media_dir": "/home/ubuntu/qwenmediafiles",
}


def _load_from_json() -> Dict[str, str]:
    path = os.path.join(os.getcwd(), "config.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if isinstance(v, str)}
    except Exception:
        return {}


def _compose() -> Dict[str, str]:
    cfg = {**DEFAULTS, **_load_from_json()}
    # env overrides
    if os.getenv("VLLM_BASE_URL"):
        cfg["base_url"] = os.getenv("VLLM_BASE_URL")
    if os.getenv("VLLM_API_KEY"):
        cfg["api_key"] = os.getenv("VLLM_API_KEY")
    if os.getenv("VLLM_MODEL"):
        cfg["model"] = os.getenv("VLLM_MODEL")
    if os.getenv("VLLM_MEDIA_DIR"):
        cfg["media_dir"] = os.getenv("VLLM_MEDIA_DIR")
    return cfg


_CFG = _compose()

VLLM_BASE_URL: str = _CFG["base_url"]
VLLM_API_KEY: str = _CFG["api_key"]
VLLM_MODEL: str = _CFG["model"]
MEDIA_DIR: str = _CFG["media_dir"]
