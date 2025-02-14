import logging

import numpy as np
from ortools.linear_solver import pywraplp
from pandas.core.arrays import ExtensionArray

logger = logging.getLogger(__name__)


def compare_opt_to_random(
    y: np.ndarray | ExtensionArray,
    y_pred: np.ndarray | ExtensionArray,
    y_error_pred: np.ndarray | ExtensionArray,
    rns: np.random.RandomState,
    pcts: list[float],
    n_tries=100,
    n_samples=100,
) -> list[dict]:
    report = []
    chs = [rns.choice(y.shape[0], size=n_samples, replace=True) for _ in range(n_tries)]
    randoms = [np.mean(y[ch]) for ch in chs]
    m, s = np.mean(randoms), np.std(randoms)

    for pct in pcts:
        n_select = int(pct * y.shape[0])
        if n_select < 10:
            logger.warning(f"for pct {pct} n_select is {n_select}")
        x_opt = find_opt(y_pred, y_error_pred, n_select=n_select)
        y_opt = (y * x_opt).sum() / x_opt.sum()
        y_opt_pred = (y_pred * x_opt).sum() / x_opt.sum()
        report += [
            {
                "li_opt": y_opt,
                "perf_type": "gt",
                "li_random_mean": m,
                "li_random_std": s,
                "pct": pct,
            },
            {
                "li_opt": y_opt_pred,
                "perf_type": "pred",
                "li_random_mean": m,
                "li_random_std": s,
                "pct": pct,
            },
        ]

    return report


def find_opt(
    exp_returns: np.ndarray | ExtensionArray,
    exp_sigmas: np.ndarray | ExtensionArray,
    n_select: int = 20,
    ratio_return_sigma: float = 1.0,
):
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not ratio_return_sigma > 0:
        raise ValueError("ratio_return_sigma should be positive")
    n = len(exp_returns)
    objective_sigma = []
    obj_returns = []
    obj_number = []

    for i in range(n):
        x = solver.IntVar(0, 1, "var")
        objective_sigma.append(x * exp_sigmas[i])
        obj_returns.append(x * exp_returns[i])
        obj_number.append(x)
    solver.Add(solver.Sum(obj_number) <= n_select)
    solver.Maximize(
        ratio_return_sigma * solver.Sum(obj_returns) - solver.Sum(objective_sigma)
    )

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        return np.array([v.solution_value() for v in solver.variables()])
    else:
        raise ValueError("Solver failed")
