"""给 RAGFlow 租户绑定通义 embedding，并重解析已上传 PDF。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

B2C_ROOT = Path(__file__).resolve().parents[1]
UVGRAPH_ROOT = B2C_ROOT.parent
sys.path[:0] = [str(UVGRAPH_ROOT), str(B2C_SRC := B2C_ROOT / "src")]

INNER = r"""
from api.utils.crypt import crypt
import json, os, urllib.request, urllib.error, urllib.parse
BASE='http://127.0.0.1:9380'
EMAIL='b2c@ragflow.io'
PASSWORD='b2c_ragflow_2026'
FACTORY='Tongyi-Qianwen'
EMBD='text-embedding-v3'
KEY=os.environ['DASHSCOPE_API_KEY']

def call(method, path, auth='', body=None):
    data=None if body is None else json.dumps(body).encode()
    headers={'Accept':'application/json'}
    if auth: headers['Authorization']=auth
    if body is not None: headers['Content-Type']='application/json'
    req=urllib.request.Request(BASE+path, data=data, headers=headers, method=method)
    try:
        resp=urllib.request.urlopen(req, timeout=120)
        raw=resp.read().decode()
        h=resp.headers.get('Authorization') or auth
        payload=json.loads(raw) if raw else {}
        payload['_auth']=h
        return payload
    except urllib.error.HTTPError as e:
        return json.loads(e.read().decode('utf-8','replace') or '{}')

cipher=crypt(PASSWORD)
login=call('POST','/api/v1/auth/login', body={'email':EMAIL,'password':cipher})
auth=login.get('_auth') or ''
print('login', login.get('code'), bool(auth), login.get('message'))
print('add', call('PUT','/api/v1/providers', auth, {'provider_name':FACTORY}))
print('inst', call('POST','/api/v1/providers/'+urllib.parse.quote(FACTORY)+'/instances', auth, {
    'instance_name':'tongyi','api_key':KEY,
    'base_url':'https://dashscope.aliyuncs.com/compatible-mode/v1',
    'model_info':[{'model_type':['embedding'],'model_name':EMBD}],
}))
print('def', call('PATCH','/api/v1/models/default', auth, {
    'model_type':'embedding','model_provider':FACTORY,'model_instance':'tongyi','model_name':EMBD,
}))
"""


def main() -> None:
    from src.configs import get_app_settings

    key = (get_app_settings().dashscope_api_key or "").strip()
    if not key:
        raise SystemExit("未配置 DASHSCOPE_API_KEY")
    out = subprocess.check_output(
        ["docker", "exec", "-e", f"DASHSCOPE_API_KEY={key}", "docker-ragflow-cpu-1", "/ragflow/.venv/bin/python", "-c", INNER],
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    print(out)
    from algorithm.ragflow_kb import _request, _rows, _wait_parse, ensure_dataset

    ds = ensure_dataset()
    ds_id = str(ds.get("id") or "")
    upd = _request("PUT", f"/api/v1/datasets/{ds_id}", json_body={"embedding_model": "text-embedding-v3@Tongyi-Qianwen"})
    print("dataset_embd", upd.get("code"), upd.get("message"))
    docs = _rows(_request("GET", f"/api/v1/datasets/{ds_id}/documents", query={"page": 1, "page_size": 20}))
    ids = [str(d.get("id")) for d in docs if d.get("id")]
    print("docs", [(d.get("name"), d.get("run"), d.get("chunk_count")) for d in docs])
    if ids:
        parsed = _request("POST", f"/api/v1/datasets/{ds_id}/chunks", json_body={"document_ids": ids})
        print("parse", parsed.get("code"), parsed.get("message"))
        print("wait", {k: _wait_parse(ds_id, ids[0]).get(k) for k in ("run", "chunk_count")})


if __name__ == "__main__":
    main()
