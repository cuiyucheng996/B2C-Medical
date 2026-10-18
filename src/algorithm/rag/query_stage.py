"""检索前：多查询扩展（规则，不调 LLM）。HyDE/改写留配置位。"""

from __future__ import annotations

from algorithm.rag.types import RagPipelineConfig

__all__ = ["expand_queries"]


def expand_queries(
    query: str,
    seeds: list[str],
    *,
    config: RagPipelineConfig,
) -> list[str]:
    base = (query or "").strip()
    if not base:
        return []
    if not config.enhanced or not config.multi_query:
        return [base]
    out: list[str] = [base]
    for seed in seeds:
        token = str(seed or "").strip()[:80]
        if token and token != base:
            out.append(token)
    compact = base.replace("请问", "").replace("帮我查", "").strip()
    if compact and compact != base and len(compact) >= 4:
        out.append(compact)
    dedup: list[str] = []
    seen: set[str] = set()
    for item in out:
        if item not in seen:
            seen.add(item)
            dedup.append(item)
    # 每个实体单独一路，不被 max_queries 把后面的实体裁掉；总路数最多 6。
    keep = max(1, config.max_queries, min(len(dedup), 6))
    return dedup[:keep]
