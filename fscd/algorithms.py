from __future__ import annotations

from collections.abc import Callable

import numpy as np

from fscd.graphs import pdag_to_dag_adjacency

SUPPORTED_ALGORITHMS = ("pc", "ges", "boss")


def _scalarize(value: object) -> float:
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        raise ValueError(f"Expected a scalar-compatible value, got shape {array.shape}.")
    return float(array.reshape(-1)[0])


def _patched_local_score_bic(data: np.ndarray, node: int, parents: list[int], parameters=None) -> float:
    cov = np.cov(data.T, ddof=0)
    n_samples = data.shape[0]
    lambda_value = 0.5 if parameters is None else parameters["lambda_value"]

    sigma = float(cov[node, node])
    if parents:
        yx = cov[np.ix_([node], parents)]
        xx = cov[np.ix_(parents, parents)]
        try:
            xx_inv = np.linalg.inv(xx)
        except np.linalg.LinAlgError:
            xx_inv = np.linalg.pinv(xx)
        sigma = _scalarize(cov[node, node] - yx @ xx_inv @ yx.T)

    if sigma <= 0:
        sigma = np.finfo(float).eps

    likelihood = -0.5 * n_samples * (1 + np.log(sigma))
    penalty = lambda_value * (len(parents) + 1) * np.log(n_samples)
    return float(likelihood - penalty)


def _patched_local_score_bic_from_cov(data: tuple[np.ndarray, int], node: int, parents: list[int], parameters=None) -> float:
    cov, n_samples = data
    lambda_value = 0.5 if parameters is None else parameters["lambda_value"]

    sigma = float(cov[node, node])
    if parents:
        yx = cov[np.ix_([node], parents)]
        xx = cov[np.ix_(parents, parents)]
        try:
            xx_inv = np.linalg.inv(xx)
        except np.linalg.LinAlgError:
            xx_inv = np.linalg.pinv(xx)
        sigma = _scalarize(cov[node, node] - yx @ xx_inv @ yx.T)

    if sigma <= 0:
        sigma = np.finfo(float).eps

    likelihood = -0.5 * n_samples * (1 + np.log(sigma))
    penalty = lambda_value * (len(parents) + 1) * np.log(n_samples)
    return float(likelihood - penalty)


_patched_local_score_bic.__name__ = "local_score_BIC"
_patched_local_score_bic_from_cov.__name__ = "local_score_BIC_from_cov"


def _patch_causallearn_bic_scores() -> None:
    from causallearn.score import LocalScoreFunction as local_score_module
    from causallearn.search.PermutationBased import BOSS as boss_module
    from causallearn.search.ScoreBased import GES as ges_module

    local_score_module.local_score_BIC = _patched_local_score_bic
    local_score_module.local_score_BIC_from_cov = _patched_local_score_bic_from_cov
    ges_module.local_score_BIC = _patched_local_score_bic
    ges_module.local_score_BIC_from_cov = _patched_local_score_bic_from_cov
    boss_module.local_score_BIC = _patched_local_score_bic
    boss_module.local_score_BIC_from_cov = _patched_local_score_bic_from_cov


def run_pc(samples: np.ndarray) -> np.ndarray:
    from causallearn.search.ConstraintBased.PC import pc

    graph = pc(samples, show_progress=False).G
    return pdag_to_dag_adjacency(graph)


def run_ges(samples: np.ndarray) -> np.ndarray:
    from causallearn.search.ScoreBased.GES import ges

    _patch_causallearn_bic_scores()
    graph = ges(samples)["G"]
    return pdag_to_dag_adjacency(graph)


def run_boss(samples: np.ndarray) -> np.ndarray:
    from causallearn.search.PermutationBased.BOSS import boss

    _patch_causallearn_bic_scores()
    graph = boss(samples, verbose=False)
    return pdag_to_dag_adjacency(graph)


def algorithm_registry() -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    return {
        "pc": run_pc,
        "ges": run_ges,
        "boss": run_boss,
    }


def run_algorithm(name: str, samples: np.ndarray) -> np.ndarray:
    registry = algorithm_registry()
    key = name.lower()
    if key not in registry:
        raise ValueError(f"Unsupported algorithm '{name}'. Supported algorithms: {SUPPORTED_ALGORITHMS}.")
    return registry[key](samples)
