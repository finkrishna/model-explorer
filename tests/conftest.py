import sys
from pathlib import Path

# Make the backend modules importable (they import each other by bare name).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
