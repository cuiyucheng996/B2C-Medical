"""B2C 陪聊会话：group_id/user_id = 用户；session_id = 本场 Redis short/compress + 主图 checkpoint。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

__all__ = ["CompanionSession", "SessionStore", "get_session_store"]


def _session_memory_mod():
    from configs.paths_b2c import import_b2c_algorithm

    return import_b2c_algorithm("session_memory")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class CompanionSession:
    session_id: str
    user_id: str
    created_at: str
    user: list[dict[str, Any]] = field(default_factory=list)
    last_preprocess: dict[str, Any] | None = None
    last_report: dict[str, Any] | None = None
    last_file: dict[str, Any] | None = None
    slot_memory: dict[str, list[str]] = field(default_factory=lambda: {"疾病": [], "症状": [], "部位": []})
    persisted: bool = False
    last_activity_at: str = ""

    @property
    def group_id(self) -> str:
        return self.user_id

    def append(self, *, role_type: str, content: str, role: str | None = None) -> dict[str, Any]:
        item = {
            "uuid": str(uuid4()),
            "role_type": role_type,
            "role": role or role_type,
            "content": (content or "").strip(),
            "timestamp": _utc_now(),
        }
        self.user.append(item)
        self.touch()
        return item

    def touch(self) -> None:
        self.last_activity_at = _utc_now()

    def merge_slot_memory(self, preprocess: dict[str, Any] | None) -> None:
        """本轮抽出的部位/病/症并入整场槽位，新出现的放最前。"""
        payload = preprocess if isinstance(preprocess, dict) else {}
        buckets = self.slot_memory
        for key in ("疾病", "症状", "部位", "检查", "指标"):
            buckets.setdefault(key, [])
        mentions = payload.get("mentions") if isinstance(payload.get("mentions"), dict) else {}
        for key in ("疾病", "症状", "部位", "检查", "指标"):
            for name in mentions.get(key) or []:
                self._prepend_slot(key, str(name))
        for hit in payload.get("alignments") or []:
            if not isinstance(hit, dict):
                continue
            name = str(hit.get("standard_name") or hit.get("mention") or "").strip()
            label = str(hit.get("node_label") or "")
            if "Disease" in label:
                self._prepend_slot("疾病", name)
            elif "Symptom" in label:
                self._prepend_slot("症状", name)
            elif "BodyPart" in label:
                self._prepend_slot("部位", name)

    def _prepend_slot(self, key: str, name: str) -> None:
        text = (name or "").strip()
        if not text or text.startswith("user("):
            return
        items = self.slot_memory.setdefault(key, [])
        if text in items:
            items.remove(text)
        items.insert(0, text)
        del items[12:]

    def bind_user_extract(self, preprocess: dict[str, Any] | None) -> None:
        """把本轮病理抽离结果记在刚说的那句用户话上，关场按句写 Finding。"""
        payload = preprocess if isinstance(preprocess, dict) else {}
        for item in reversed(self.user):
            if item.get("role_type") != "user":
                continue
            item["mentions"] = payload.get("mentions") if isinstance(payload.get("mentions"), dict) else {}
            item["alignments"] = list(payload.get("alignments") or [])
            item["entity_extracted"] = bool(payload.get("entity_extracted"))
            return

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "group_id": self.user_id,
            "created_at": self.created_at,
            "persisted": self.persisted,
            "last_preprocess": self.last_preprocess,
            "last_report": self.last_report,
            "last_file": self.last_file,
            "slot_memory": {key: list(val) for key, val in (self.slot_memory or {}).items()},
            "user": list(self.user),
            "last_activity_at": self.last_activity_at,
        }


class SessionStore:
    """一人一场进行中的会话，按 user_id 取。"""

    def __init__(self) -> None:
        self._by_user: dict[str, CompanionSession] = {}

    def get_active_for_user(self, user_id: str) -> CompanionSession | None:
        uid = (user_id or "").strip()
        if not uid:
            return None
        return self._by_user.get(uid)

    def create(self, user_id: str, *, force: bool = False) -> CompanionSession:
        uid = (user_id or "").strip()
        if not uid:
            raise ValueError("user_id 不能为空")
        existing = self._by_user.get(uid)
        if existing is not None and not force:
            return existing
        session = CompanionSession(
            session_id=str(uuid4()),
            user_id=uid,
            created_at=_utc_now(),
            last_activity_at=_utc_now(),
        )
        sm = _session_memory_mod()
        session.slot_memory = sm.validate_slot_memory_for_session(
            session.slot_memory,
            user_id=uid,
            session_id=session.session_id,
        )
        self._by_user[uid] = session
        return session

    def create_new(self, user_id: str) -> CompanionSession:
        """显式新开一场，替换该用户进行中的会话。"""
        uid = (user_id or "").strip()
        if not uid:
            raise ValueError("user_id 不能为空")
        existing = self._by_user.get(uid)
        if existing is not None:
            sm = _session_memory_mod()
            sm.release_session_runtime_state(
                existing.user_id,
                existing.session_id,
                clear_slots=False,
                clear_checkpoint=True,
            )
        session = CompanionSession(
            session_id=str(uuid4()),
            user_id=uid,
            created_at=_utc_now(),
            last_activity_at=_utc_now(),
        )
        sm = _session_memory_mod()
        session.slot_memory = sm.validate_slot_memory_for_session(
            session.slot_memory,
            user_id=uid,
            session_id=session.session_id,
        )
        self._by_user[uid] = session
        return session

    def list_active(self) -> list[CompanionSession]:
        return list(self._by_user.values())

    def get(self, key: str) -> CompanionSession | None:
        token = (key or "").strip()
        if not token:
            return None
        hit = self._by_user.get(token)
        if hit is not None:
            return hit
        for session in self._by_user.values():
            if session.session_id == token:
                return session
        return None

    def drop(self, session: CompanionSession) -> None:
        gid = (session.user_id or "").strip()
        current = self._by_user.get(gid)
        if current is session:
            self._by_user.pop(gid, None)


_store: SessionStore | None = None


def get_session_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store
