from pathlib import Path
import sys

_EXTERNAL = Path(__file__).resolve().parents[2] / "external" / "QuasiSymmetries"

if str(_EXTERNAL) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL))

