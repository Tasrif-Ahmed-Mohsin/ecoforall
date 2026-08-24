"""pytest config — ensure `project/` is on sys.path so `import src.*` works."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
