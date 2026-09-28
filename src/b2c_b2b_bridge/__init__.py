"""兼容旧调用：逻辑已在 b2b_fork，不再切换 sys.path / sys.modules。"""

from __future__ import annotations

import b2b_fork.align_rerank as _align_rerank
import b2b_fork.api_config as _api_config
import b2b_fork.disease_plain as _disease_plain
import b2b_fork.entity_synonyms as _entity_synonyms
import b2b_fork.negation as _negation
import b2b_fork.uie_extract as _uie_extract


def b2b_entity_align():
    raise RuntimeError("entity_align 未复刻；C 端请用 algorithm.entity_align_c")


def b2b_align_rerank():
    return _align_rerank


def b2b_negation():
    return _negation


def b2b_uie_extract():
    return _uie_extract


def b2b_entity_synonyms():
    return _entity_synonyms


def b2b_intent_classify_bert():
    raise RuntimeError("intent_classify_bert 未复刻到 B2C")


def b2b_disease_plain():
    return _disease_plain


def b2b_api_config():
    return _api_config


def warmup_b2b_bridge() -> None:
    for loader in (b2b_uie_extract, b2b_negation, b2b_entity_synonyms):
        loader()
