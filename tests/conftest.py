"""Put `src` on the path once, for every test module.

Each test file used to do this itself. That works until a new file forgets, and the
failure - `ModuleNotFoundError: No module named 'penumbra'` - looks like a broken
install rather than a missing line.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
