from tqdm import tqdm

from connectome_data.constants import Simplicity, Directedness, Weightedness, tgt_dir, REAL_EDGES, RAND_DIR
from connectome_data.make_data.metrics import make_metrics, make_ensemble_metrics
from connectome_data.utils import GraphSerialiser


def combine(
    simplicity=Simplicity.SIMPLE, directedness=Directedness.DIRECTED, weightedness=Weightedness.UNWEIGHTED,
    reals=None, rands=None
):
    """

    :param simplicity:
    :param directedness:
    :param weightedness:
    :param reals: names of graphs which should not be randomised
    :param rands: names of graphs which should be randomised
    :return:
    """
    reals = reals or []
    rands = rands or []
    name = f'{"_".join(reals)}+{"_".join(rands)}'

    dpath = tgt_dir(simplicity, directedness, weightedness) / "combined" / name
    (dpath / RAND_DIR).mkdir(parents=True, exist_ok=True)

    real_gs = [GraphSerialiser(real, simplicity, directedness, weightedness) for real in reals]
    rand_gs = [GraphSerialiser(rand, simplicity, directedness, weightedness) for rand in rands]

    enorm = real_gs[0].edges_normaliser

    real_edges = enorm.union(*[enorm.csv_to_rows(r.real_csv) for r in real_gs])
    nonrand_edges = enorm.union(real_edges, *[enorm.csv_to_rows(r.real_csv) for r in rand_gs])
    enorm.rows_to_csv(nonrand_edges, dpath / REAL_EDGES)

    for idx_fpaths in tqdm(zip(*[gs.rand_paths() for gs in rand_gs]), total=1000):
        rand_edges = enorm.union(real_edges, *[enorm.csv_to_rows(fpath) for _, fpath in idx_fpaths])
        enorm.rows_to_csv(rand_edges, dpath / RAND_DIR / idx_fpaths[0][1].name)


def comb_name(reals=None, rands=None):
    reals = reals or []
    rands = rands or []
    return f'{"_".join(reals)}+{"_".join(rands)}'


def make_comb_metrics(
    simplicity=Simplicity.SIMPLE, directedness=Directedness.DIRECTED, weightedness=Weightedness.UNWEIGHTED,
    reals=None, rands=None, workers=None
):
    name = comb_name(reals, rands)

    dpath = tgt_dir(simplicity, directedness, weightedness) / "combined" / name

    make_metrics(dpath, workers)
    make_ensemble_metrics(dpath)


if __name__ == '__main__':
    reals_rands = {'reals': ['ac'], 'rands': ['monoamine']}
    combine(**reals_rands)
    make_comb_metrics(**reals_rands, workers=10)
