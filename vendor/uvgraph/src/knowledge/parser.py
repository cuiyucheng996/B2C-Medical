"""疾病知识库 markdown 解析。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .md_table import parse_markdown_table

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DISEASE_MD = PROJECT_ROOT / "data" / "rawdata" / "disease.md"

DISEASE_TABLE_COLUMNS = ("疾病", "症状类型", "包含症状", "通俗同义词")


@dataclass(frozen=True)
class DiseaseRecord:
    disease_name: str
    symptom_category: str
    symptoms: str
    synonyms: str

    def to_index_text(self) -> str:
        parts = [self.disease_name, self.symptom_category]
        if self.symptoms:
            parts.append(f"症状:{self.symptoms}")
        if self.synonyms:
            parts.append(f"同义词:{self.synonyms}")
        return " | ".join(parts)

    def to_metadata(self) -> dict[str, str]:
        return {
            "record_type": "disease",
            "disease_name": self.disease_name,
            "symptom_category": self.symptom_category,
            "symptoms": self.symptoms,
            "synonyms": self.synonyms,
        }


def _row_to_disease_record(row: dict[str, str]) -> DiseaseRecord | None:
    disease_name = row.get("疾病", "").strip()
    if not disease_name:
        return None
    return DiseaseRecord(
        disease_name=disease_name,
        symptom_category=row.get("症状类型", "").strip(),
        symptoms=row.get("包含症状", "").strip(),
        synonyms=row.get("通俗同义词", "").strip(),
    )


def parse_markdown_records(
    path: Path,
    *,
    row_mapper: Callable[[dict[str, str]], DiseaseRecord | None],
    required_columns: tuple[str, ...] = (),
) -> list[DiseaseRecord]:
    """通用入口：先解析 md 表，再用业务 mapper 转 Record。"""
    table_rows = parse_markdown_table(path)
    if required_columns:
        headers = set(table_rows[0])
        missing = [col for col in required_columns if col not in headers]
        if missing:
            msg = f"missing columns {missing} in {path}"
            raise ValueError(msg)

    records: list[DiseaseRecord] = []
    for row in table_rows:
        record = row_mapper(row)
        if record is not None:
            records.append(record)

    if not records:
        msg = f"no records mapped from {path}"
        raise ValueError(msg)

    return records


def parse_disease_markdown(path: Path | None = None) -> list[DiseaseRecord]:
    """解析疾病表 disease.md。"""
    md_path = path or DEFAULT_DISEASE_MD
    return parse_markdown_records(
        md_path,
        row_mapper=_row_to_disease_record,
        required_columns=DISEASE_TABLE_COLUMNS,
    )
