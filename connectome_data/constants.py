from enum import Enum
from pathlib import Path

PACKAGE_ROOT = Path(__file__).absolute().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

SRC_ROOT = PROJECT_ROOT / "source_data"
EMMONS_ROOT = SRC_ROOT / "emmons"
BENTLEY_ROOT = SRC_ROOT / "bentley_2016_S1_dataset" / "S1 Dataset. Included are edge lists and source data for monoamine and neuropeptide networks"
AC_ROOT = SRC_ROOT / "connectome_construct" / "physical" / "src_data"

TGT_ROOT = PROJECT_ROOT / "target_data"
UNDIR_EDGELISTS_ROOT = TGT_ROOT / "undir_simple_edgelists"

SWAPS_PER_EDGE = 10
N_RANDOM = 1000


class EdgeType(Enum):
    CHEMICAL = "chemical"
    ELECTRICAL = "electrical"
    MONOAMINE = "monoamine"
    NEUROPEPTIDE = "neuropeptide"

    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        if isinstance(other, str):
            return str(self) == other
        else:
            return super().__eq__(other)

    def __hash__(self):
        return hash(self.value)
