from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

_SRC = Path(__file__).resolve().parents[1] / "src"
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

from algorithm.kb_eval import run_kb_eval, run_kb_tool_probe


class KbEvalTest(unittest.TestCase):
    @mock.patch("algorithm.kb_eval.finish_preprocess_after_intent")
    @mock.patch("algorithm.intent_classify._default_chat_model", return_value=None)
    def test_part_query_calls_graph(self, _llm: mock.MagicMock, finish: mock.MagicMock) -> None:
        finish.return_value = mock.Mock(
            mentions={"部位": ["头部"]},
            alignments=[
                {
                    "mention": "头部",
                    "standard_name": "头面部",
                    "node_label": "ConsultBodyPart",
                    "method": "synonym",
                }
            ],
        )
        with mock.patch(
            "algorithm.kb_eval.query_consult_graph",
            return_value={"items": ["偏头痛"], "summary": "ok"},
        ) as graph, mock.patch("algorithm.kb_eval.retrieve_hits") as rag:
            row = run_kb_eval(query="头部常见有哪些病")
        graph.assert_called_once()
        self.assertEqual(graph.call_args.kwargs["alignments"][0]["standard_name"], "头面部")
        rag.assert_not_called()
        self.assertEqual(row["handler"], "kg_part_diseases")
        self.assertEqual(row["channel"], "neo4j")
        self.assertEqual(row["tools"], ["query_consult_graph"])
        self.assertEqual(row["mentions"]["部位"], ["头部"])
        self.assertFalse(row["milvus"]["queried"])
        self.assertIn("未查询", row["milvus"]["detail"])

    def test_part_seed_from_alignment(self) -> None:
        from algorithm.kg_query import _resolve_seeds

        seeds = _resolve_seeds(
            "kg_part_diseases",
            "头部常见有哪些病",
            {"部位": ["头部"]},
            [{"node_label": "ConsultBodyPart", "standard_name": "头面部", "mention": "头部"}],
        )
        self.assertEqual(seeds[0], "头面部")

    def test_graph_query_uses_aligned_entry(self) -> None:
        with (
            mock.patch("algorithm.kg_query.ensure_uvgraph_root"),
            mock.patch(
                "src.RAGgraph.driver.neo4j_driver.get_neo4j_driver",
                return_value=mock.MagicMock(),
            ),
            mock.patch("src.configs.NEO4J_CONFIG", {"database": "neo4j"}),
            mock.patch("algorithm.kg_query._names_for_anchor", return_value=["头痛"]) as cypher,
        ):
            from algorithm.kg_query import query_consult_graph

            row = query_consult_graph(
                handler="kg_part_diseases",
                query="头部常见有哪些病",
                mentions={"部位": ["头部"]},
                alignments=[
                    {
                        "mention": "头部",
                        "standard_name": "头面部",
                        "node_label": "ConsultBodyPart",
                        "method": "synonym",
                    }
                ],
            )
        self.assertEqual(cypher.call_args.kwargs["seeds"][0], "头面部")
        self.assertEqual(row["items"], ["头痛"])
        self.assertEqual(row["alignments"][0]["standard_name"], "头面部")

    @mock.patch("algorithm.kb_eval.finish_preprocess_after_intent")
    @mock.patch("algorithm.intent_classify._default_chat_model", return_value=None)
    def test_exam_query_calls_ragflow(self, _llm: mock.MagicMock, finish: mock.MagicMock) -> None:
        finish.return_value = mock.Mock(mentions={}, alignments=[])
        packed = {"items": ["其它项目说明", "血常规含白细胞"], "summary": "ok", "seeds": ["血常规"]}
        with mock.patch("algorithm.kb_eval.search_exam_handbook", return_value=packed) as handbook:
            row = run_kb_eval(query="血常规这个检查是查什么的")
        handbook.assert_called_once()
        self.assertEqual(handbook.call_args.kwargs["handler"], "kg_exam_meaning")
        self.assertEqual(row["handler"], "kg_exam_meaning")
        self.assertEqual(row["channel"], "ragflow")
        self.assertEqual(row["rerank"][0]["text"], "血常规含白细胞")

    @mock.patch("algorithm.kb_eval.finish_preprocess_after_intent")
    def test_tool_probe_invokes_selected(self, finish: mock.MagicMock) -> None:
        finish.return_value = mock.Mock(
            resolved_text="头部常见有哪些病",
            mentions={"部位": ["头部"]},
            alignments=[],
            negated=[],
            entity_extracted=True,
            intent_handler="kg_part_diseases",
        )
        fake_env = '{"ok":true,"usable":true,"summary":"ok","data":{"items":["a"]}}'
        with mock.patch(
            "algorithm.kb_eval._KB_TOOL_MAP",
            {
                "query_consult_graph": mock.Mock(
                    name="query_consult_graph", invoke=mock.Mock(return_value=fake_env)
                )
            },
        ):
            row = run_kb_tool_probe(query="头部常见有哪些病", tools=["query_consult_graph"])
        self.assertTrue(row["ok"])
        self.assertEqual(row["tool_results"][0]["tool"], "query_consult_graph")
        self.assertEqual(row["mentions"]["部位"], ["头部"])


if __name__ == "__main__":
    unittest.main()
