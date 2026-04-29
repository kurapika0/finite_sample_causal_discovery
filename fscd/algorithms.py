from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

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


@lru_cache(maxsize=1)
def _load_pytetrad_modules() -> tuple[object, object, object]:
    try:
        import importlib.resources as importlib_resources
        import jpype
        import jpype.imports  # noqa: F401
        import pytetrad  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised in integration use
        raise ImportError(
            "py-tetrad PC requires the 'JPype1' and 'py-tetrad' packages to be installed."
        ) from exc

    jar_path = str(importlib_resources.files("pytetrad").joinpath("resources", "tetrad-current.jar"))
    if not jpype.isJVMStarted():
        try:
            jpype.startJVM(jpype.getDefaultJVMPath(), classpath=[jar_path])
        except OSError as exc:  # pragma: no cover - exercised in integration use
            raise RuntimeError(
                "Failed to start the JVM for py-tetrad. Ensure a compatible JDK is installed."
            ) from exc

    import pytetrad.tools.translate as tetrad_translate
    import edu.cmu.tetrad.search as tetrad_search
    import edu.cmu.tetrad.search.test as tetrad_test

    return tetrad_translate, tetrad_search, tetrad_test


def _tetrad_graph_to_matrix(graph: object, node_names: list[str]) -> np.ndarray:
    endpoint_code = {"NULL": 0, "TAIL": -1, "ARROW": 1, "CIRCLE": 2}
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    matrix = np.zeros((len(node_names), len(node_names)), dtype=int)

    for edge in list(graph.getEdges()):
        left_name = str(edge.getNode1().getName())
        right_name = str(edge.getNode2().getName())
        left_endpoint = edge.getEndpoint1().name()
        right_endpoint = edge.getEndpoint2().name()

        if left_endpoint not in endpoint_code or right_endpoint not in endpoint_code:
            raise ValueError(
                f"Unsupported Tetrad edge endpoints ({left_endpoint}, {right_endpoint}) "
                f"between {left_name} and {right_name}."
            )

        left = name_to_index[left_name]
        right = name_to_index[right_name]
        matrix[right, left] = endpoint_code[left_endpoint]
        matrix[left, right] = endpoint_code[right_endpoint]

    return matrix


def run_pc(samples: np.ndarray) -> np.ndarray:
    import pandas as pd

    tetrad_translate, tetrad_search, tetrad_test = _load_pytetrad_modules()

    samples = np.asarray(samples, dtype=float)
    node_names = [f"X{i}" for i in range(samples.shape[1])]
    data_frame = pd.DataFrame(samples, columns=node_names)
    data = tetrad_translate.pandas_data_to_tetrad(data_frame)

    independence_test = tetrad_test.IndTestFisherZ(data, 0.05)
    algorithm = tetrad_search.Pc(independence_test)
    algorithm.setFasStable(True)
    algorithm.setColliderOrientationStyle(tetrad_search.Pc.ColliderOrientationStyle.MAX_P)
    algorithm.setAllowBidirected(tetrad_search.Pc.AllowBidirected.DISALLOW)
    algorithm.setForbidDirectedCycles(True)
    algorithm.setMeekCycleSafe(True)

    graph = algorithm.search()
    matrix = _tetrad_graph_to_matrix(graph, node_names)
    return pdag_to_dag_adjacency(matrix)


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
