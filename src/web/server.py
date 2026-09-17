"""纠错 Agent 本地测试服务。"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agent.fixtures import SPELL_CHECK_SAMPLES
from agent.spell_check_agent import correct_text

WEB_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Spell Check Agent Test")


class CorrectRequest(BaseModel):
    text: str = Field(min_length=1, description="待纠错文本")


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.post("/api/correct")
def correct(req: CorrectRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text 不能为空")
    result = correct_text(text)
    return result.model_dump()


@app.get("/api/presets")
def presets():
    return [
        {
            "id": sample.id,
            "name": sample.name,
            "text": sample.text,
            "expect": sample.expect,
        }
        for sample in SPELL_CHECK_SAMPLES
    ]
