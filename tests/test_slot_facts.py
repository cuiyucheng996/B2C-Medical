from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from algorithm.fact_extract import facts_from_slot_memory, links_from_slot_memory


class SlotMemoryFactsTest(unittest.TestCase):
    def test_slots_become_facts_and_links(self) -> None:
        slots = {"部位": ["胃"], "疾病": ["胃炎"], "症状": ["胃痛"]}
        facts = facts_from_slot_memory(slots)
        names = {item["name"] for item in facts}
        self.assertEqual(names, {"胃", "胃炎", "胃痛"})
        self.assertTrue(all("用户提到" in item["fact"] for item in facts))
        links = links_from_slot_memory(slots)
        kinds = {item["kind"] for item in links}
        self.assertEqual(kinds, {"body_part", "disease", "symptom"})

    def test_mentions_merge_without_dup(self) -> None:
        facts = facts_from_slot_memory(
            {"部位": ["胃"]},
            {"部位": ["胃"], "症状": ["反酸"]},
        )
        names = [item["name"] for item in facts]
        self.assertEqual(names.count("胃"), 1)
        self.assertIn("反酸", names)

    def test_alignments_to_links(self) -> None:
        from algorithm.fact_extract import links_from_alignments

        links = links_from_alignments(
            [{"standard_name": "胃", "node_label": "ConsultBodyPart", "mention": "胃这儿"}]
        )
        self.assertEqual(links[0]["kind"], "body_part")
        self.assertEqual(links[0]["name"], "胃")


class TalkEntitiesTest(unittest.TestCase):
    def test_person_and_event_from_chat(self) -> None:
        from algorithm.talk_entities import extract_talk_entities

        rows = extract_talk_entities("\u6628\u5929\u8ddf\u5f20\u4e09\u4e00\u8d77\u5403\u996d\uff0c\u6211\u5988\u8bf4\u522b\u71ac\u591c")
        kinds = {item["kind"] for item in rows}
        names = {item["name"] for item in rows}
        self.assertIn("person", kinds)
        self.assertIn("\u5f20\u4e09", names)
        self.assertTrue(any(item["kind"] == "event" and "\u5403\u996d" in item["name"] for item in rows))
        self.assertIn("\u5988\u5988", names)

    def test_thing_bought(self) -> None:
        from algorithm.talk_entities import extract_talk_entities

        rows = extract_talk_entities("\u6211\u4e70\u4e86\u52a0\u6e7f\u5668")
        self.assertTrue(any(item["kind"] == "thing" and item["name"] == "\u52a0\u6e7f\u5668" for item in rows))


class EntityMemoryMiddlewareTest(unittest.TestCase):
    def test_inject_once_per_turn(self) -> None:
        from unittest.mock import patch

        from langchain_core.messages import HumanMessage

        from middleware.context import CompanionRunContext
        from middleware.entity_memory import CompanionEntityMemoryMiddleware

        class _Runtime:
            def __init__(self, ctx):
                self.context = ctx

        query = "\u6628\u5929\u8ddf\u5f20\u4e09\u4e00\u8d77\u5403\u996d\uff0c\u6211\u4e70\u4e86\u52a0\u6e7f\u5668"
        note = "\u3010\u5b9e\u4f53\u8bb0\u5fc6\u3011\n\u672c\u8f6e\u5b9e\u4f53\uff1a\u5f20\u4e09\u3001\u52a0\u6e7f\u5668"
        mw = CompanionEntityMemoryMiddleware()
        runtime = _Runtime(CompanionRunContext(query=query, user_id="u1"))
        state = {"messages": [HumanMessage(content=query)]}
        with patch("middleware.entity_memory.recall_and_update", return_value=note):
            out = mw.before_model(state, runtime)
        self.assertIsNotNone(out)
        self.assertIn("\u5f20\u4e09", str(out["messages"][-1].content))
        self.assertIn("\u52a0\u6e7f\u5668", str(out["messages"][-1].content))
        with patch("middleware.entity_memory.recall_and_update") as mocked:
            again = mw.before_model({"messages": list(out["messages"])}, runtime)
            mocked.assert_not_called()
        self.assertIsNone(again)

    def test_no_entity_no_inject(self) -> None:
        from unittest.mock import patch

        from langchain_core.messages import HumanMessage

        from middleware.context import CompanionRunContext
        from middleware.entity_memory import CompanionEntityMemoryMiddleware

        class _Runtime:
            def __init__(self, ctx):
                self.context = ctx

        mw = CompanionEntityMemoryMiddleware()
        runtime = _Runtime(CompanionRunContext(query="\u4eca\u5929\u5929\u6c14\u4e0d\u9519"))
        with patch("middleware.entity_memory.recall_and_update", return_value=""):
            self.assertIsNone(
                mw.before_model({"messages": [HumanMessage(content="\u4eca\u5929")]}, runtime)
            )


class EntityKgRecallTest(unittest.TestCase):
    def test_recall_calls_entity_and_kg(self) -> None:
        from unittest.mock import MagicMock, patch

        from algorithm.entity_kg_memory import clear_entity_kg_memory, recall_and_update

        clear_entity_kg_memory()
        entity = MagicMock()
        entity.entity_cache = ["\u5f20\u4e09"]
        entity.load_memory_variables.return_value = {"entities": {"\u5f20\u4e09": "ate"}}
        kg = MagicMock()
        kg.load_memory_variables.return_value = {"history": "On \u5f20\u4e09: ate."}
        with patch("algorithm.entity_kg_memory._llm", return_value=object()), patch(
            "algorithm.entity_kg_memory._pair", return_value=(entity, kg)
        ):
            note = recall_and_update("hi", [], user_key="u-test")
        entity.load_memory_variables.assert_called_once()
        kg.load_memory_variables.assert_called_once()
        entity.save_context.assert_called_once()
        kg.save_context.assert_called_once()
        self.assertIn("\u5f20\u4e09", note)
        self.assertIn("ate", note)


if __name__ == "__main__":
    unittest.main()
