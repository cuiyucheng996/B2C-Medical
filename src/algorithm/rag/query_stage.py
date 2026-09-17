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
    seed_line = " ".join(str(s).strip() for s in seeds if str(s).strip())[:200]
    if seed_line and seed_line != base:
        out.append(seed_line)
    compact = base.replace("请问", "").replace("帮我查", "").strip()
    if compact and compact != base and len(compact) >= 4:
        out.append(compact)
    dedup: list[str] = []
    seen: set[str] = set()
    for item in out:
        if item not in seen:
            seen.add(item)
            dedup.append(item)
    return dedup[: max(1, config.max_queries)]
