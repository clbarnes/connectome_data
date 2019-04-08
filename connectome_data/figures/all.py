import logging

from connectome_data.figures.constants import OUT_DIR
from connectome_data.utils import TqdmStream
from connectome_data.figures.degree_dist import main as deg_distr
from connectome_data.figures.ma_distance import main as ma_dista
from connectome_data.figures.network_metrics import ac_vs_ww, phys_vs_phys_ma
from connectome_data.figures.hiveplots.phys import main as phys_hive
from connectome_data.figures.hiveplots.ma import main as ma_hive
from connectome_data.figures.module_members import main as modules

logger = logging.getLogger()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, stream=TqdmStream)

    logger.info("deg distrib")
    deg_distr(OUT_DIR / "deg_distr.svg")

    logger.info("ma distance")
    ma_dista(OUT_DIR / "ma_dista.svg")

    logger.info("ac vs ww metrics")
    ac_vs_ww(save=OUT_DIR / "ac_vs_ww.svg")

    logger.info("phys vs phys-ma metrics")
    phys_vs_phys_ma(save=OUT_DIR / "phys_vs_phys_ma.svg")

    logger.info("phys hive")
    phys_hive(OUT_DIR / "phys_hive.svg")

    logger.info("ma hive")
    ma_hive(OUT_DIR / "ma_hive.svg")

    logger.info("modules")
    modules(save=OUT_DIR / "modules.svg")

    logger.info("DONE")
