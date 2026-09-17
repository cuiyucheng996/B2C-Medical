from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from algorithm.rag.fuse import rrf_merge
from algorithm.rag.query_stage import expand_queries
from algorithm.rag.types import RagHit, RagPipelineConfig


class RagPipelineTest(unittest.TestCase):
    def test_expand_queries_multi(self) -> None:
        cfg = RagPipelineConfig(enhanced=True, multi_query=True, max_queries=3)
        qs = expand_queries("血常规这个检查是查什么的", ["血常规"], config=cfg)
        self.assertGreaterEqual(len(qs), 2)

    def test_rrf_merge_dedup(self) -> None:
        a = [RagHit(text="alpha", score=0.9, recall_rank=1, channel="handbook")]
        b = [RagHit(text="alpha", score=0.5, recall_rank=2, channel="handbook"), RagHit(text="beta", score=0.8, recall_rank=1, channel="handbook")]
        merged = rrf_merge([a, b], k=60)
        texts = [h.text for h in merged]
        self.assertEqual(texts[0], "alpha")
        self.assertIn("beta", texts)


if __name__ == "__main__":
    unittest.main()
