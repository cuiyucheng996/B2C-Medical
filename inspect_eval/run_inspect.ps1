# C 端 Inspect AI 测评（需先 uv sync；陪聊 API 可选，图任务走进程内 LangGraph）
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$task = if ($args.Count -gt 0) { $args[0] } else { "intent_gold" }
$limit = if ($args.Count -gt 1) { $args[1] } else { "" }

$py = @"
import json, sys
from pathlib import Path
import os
os.environ.setdefault('VECTOR_STORE_BACKEND', 'chroma')
root = Path(r'$Root')
sys.path[:0] = [str(root / 'src'), str(root)]
from configs.paths_b2c import ensure_b2c_runtime_path
ensure_b2c_runtime_path()
from algorithm.inspect_eval_pipeline import run_inspect_eval
lim = int('$limit') if '$limit'.strip() else None
out = run_inspect_eval('$task', limit=lim)
print(json.dumps(out, ensure_ascii=False, indent=2))
sys.exit(0 if out.get('ok') else 1)
"@

uv run python -c $py
