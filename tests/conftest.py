import sys
from pathlib import Path

# Ensure root directory is on sys.path to allow importing fmsim
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
