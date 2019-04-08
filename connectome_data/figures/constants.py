from pathlib import Path

from connectome_data.constants import Simplicity, Directedness, Weightedness, Wiring, tgt_dir

SIMPLICITY = Simplicity.SIMPLE
DIRECTEDNESS = Directedness.DIRECTED
WEIGHTEDNESS = Weightedness.UNWEIGHTED

TGT_DIR = tgt_dir(SIMPLICITY, DIRECTEDNESS, WEIGHTEDNESS)

DEFAULT_WIRING = Wiring.ALBERTSON_CHKLOVSKII

OUT_DIR = Path(__file__).absolute().parent / "out"
