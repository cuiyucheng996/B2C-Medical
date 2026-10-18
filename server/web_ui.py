"""各专属服务根路径托管 web-frontend/dist 多页构建产物。"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

_DIST = Path(__file__).resolve().parents[1] / "web-frontend" / "dist"


def mount_b2c_web_ui(app: FastAPI, entry_html: str) -> None:
    assets = _DIST / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="b2c_web_assets")

    @app.get("/", response_class=HTMLResponse)
    def _ui_index() -> HTMLResponse:
        page = _DIST / entry_html
        if not page.is_file():
            return HTMLResponse(
                status_code=503,
                content=(
                    "<html><body style='font-family:sans-serif;padding:24px'>"
                    "<h1>前端未构建</h1>"
                    "<p>请在 <code>web-frontend</code> 执行：<code>npm install</code>、<code>npm run build</code></p>"
                    f"<p>缺少：{page}</p></body></html>"
                ),
            )
        return HTMLResponse(page.read_text(encoding="utf-8"))
