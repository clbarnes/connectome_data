from pathlib import Path

PACKAGE_ROOT = Path(__file__).absolute()
PROJECT_ROOT = PACKAGE_ROOT.parent

DATA_ROOT = PROJECT_ROOT / "source_data"
EMMONS_ROOT = DATA_ROOT / "emmons"
BENTLEY_ROOT = DATA_ROOT / "bentley_2016_S1_dataset" / "S1 Dataset. Included are edge lists and source data for monoamine and neuropeptide networks"

