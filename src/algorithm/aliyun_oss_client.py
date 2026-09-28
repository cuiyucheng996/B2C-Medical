"""阿里云 OSS 预签名上传（报告 PDF 等，不经业务服务器落盘）。"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from configs.paths_b2c import ensure_uvgraph_root

logger = logging.getLogger(__name__)

__all__ = ["oss_config", "oss_presign_put"]


def _yaml_aliyun() -> dict[str, Any]:
    ensure_uvgraph_root()
    from src.configs import get_yaml_config

    block = get_yaml_config().get("mcp")
    if not isinstance(block, dict):
        return {}
    aliyun = block.get("aliyun")
    return dict(aliyun) if isinstance(aliyun, dict) else {}


def oss_config() -> dict[str, Any]:
    cfg = _yaml_aliyun()
    return {
        "bucket": os.getenv("OSS_BUCKET") or str(cfg.get("oss_bucket") or "").strip(),
        "endpoint": os.getenv("OSS_ENDPOINT") or str(cfg.get("oss_endpoint") or "oss-cn-beijing.aliyuncs.com").strip(),
        "prefix": str(cfg.get("oss_key_prefix") or "b2c-reports/").strip(),
        "has_ak": bool(
            (os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID") or os.getenv("ALIYUN_ACCESS_KEY_ID"))
            and (os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET") or os.getenv("ALIYUN_ACCESS_KEY_SECRET"))
        ),
    }


def oss_presign_put(
    filename: str,
    *,
    content_type: str = "application/pdf",
    expires: int = 3600,
) -> dict[str, Any]:
    name = (filename or "report.pdf").strip().replace("\\", "/").split("/")[-1]
    cfg = oss_config()
    bucket = cfg["bucket"]
    if not bucket:
        return {"success": False, "error": "请配置 OSS_BUCKET 或 mcp.aliyun.oss_bucket"}
    ak = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID") or os.getenv("ALIYUN_ACCESS_KEY_ID") or ""
    sk = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET") or os.getenv("ALIYUN_ACCESS_KEY_SECRET") or ""
    if not ak or not sk:
        return {"success": False, "error": "缺少阿里云 AccessKey"}
    key = f"{cfg['prefix'].rstrip('/')}/{uuid.uuid4().hex[:12]}_{name}"
    try:
        import oss2

        auth = oss2.Auth(ak, sk)
        bucket_obj = oss2.Bucket(auth, f"https://{cfg['endpoint']}", bucket)
        url = bucket_obj.sign_url("PUT", key, max(60, min(int(expires), 86400)), headers={"Content-Type": content_type})
        return {
            "success": True,
            "bucket": bucket,
            "object_key": key,
            "upload_url": url,
            "content_type": content_type,
            "hint": "客户端 PUT 文件到 upload_url，完成后把 object_key 交给后续解读流程",
        }
    except Exception as exc:
        logger.warning("oss presign failed", exc_info=True)
        return {"success": False, "error": str(exc)[:500]}
