from __future__ import annotations
import json, sys, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from penumbra.config import reactor_api_key, RUNS_DIR

req = urllib.request.Request("https://api.reactor.inc/models",
                             headers={"Reactor-API-Key": reactor_api_key()})
with urllib.request.urlopen(req, timeout=30) as r:
    models = json.loads(r.read())
out = RUNS_DIR / "_probe"; out.mkdir(parents=True, exist_ok=True)
(out / "models.json").write_text(json.dumps(models, indent=2), encoding="utf-8")
print(f"{len(models)} models\n")
for m in sorted(models, key=lambda x: x["name"]):
    gpu = f"{m.get('gpu_type','-')}x{m.get('gpu_count',0)}" if m.get("is_gpu") else "cpu"
    print(f"{m['name']:<45} {m['status']:<8} {gpu:<18} owned={m.get('is_owned')}")
    d = (m.get("description") or "").strip().replace("\n", " ")
    if d:
        print(f"    {d[:300]}")
