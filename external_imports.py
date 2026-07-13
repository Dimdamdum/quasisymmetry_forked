from pathlib import Path
import sys

_EXTERNAL = Path(__file__).resolve().parent / "external" / "QuasiSymmetries" / "src"

if str(_EXTERNAL) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL))


from quasisymmetries import Clifford, taper_hamiltonian
from quasisymmetries.state_utils import get_cisd_gs, get_hf_occ, get_hf_wfn
from quasisymmetries.bs.beam import beam_search_symmetries, BeamSearch_Symmetries
from quasisymmetries.bs.utils import mask_to_qubit_operator
from quasisymmetries.metrics import variance
from fcidump_openfermion import molecular_data_from_fcidump
