"""Download the first DROID shard (data + 3 camera videos) from nvidia/Cosmos3-DROID."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from penumbra.config import DATA_DIR
from huggingface_hub import hf_hub_download

REPO = "nvidia/Cosmos3-DROID"
FILES = [
    "success/data/chunk-000/file-000.parquet",
    "success/meta/tasks.parquet",
    "success/meta/episodes/chunk-000/file-000.parquet",
    "success/videos/observation.image.exterior_image_1_left/chunk-000/file-000.mp4",
    "success/videos/observation.image.exterior_image_2_left/chunk-000/file-000.mp4",
    "success/videos/observation.image.wrist_image_left/chunk-000/file-000.mp4",
]
cache = DATA_DIR / "hf"
cache.mkdir(parents=True, exist_ok=True)
for f in FILES:
    t0 = time.time()
    p = hf_hub_download(REPO, f, repo_type="dataset", cache_dir=str(cache))
    print(f"{f} -> {Path(p).stat().st_size/1e6:.1f} MB in {time.time()-t0:.1f}s", flush=True)
print("DONE")
