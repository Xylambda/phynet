"""Input generator.

From normally distributed returns,
create their exponential weighted average.


References
----------
- https://pandas.pydata.org/docs/user_guide/window.html#window-exponentially-weighted
- https://numpy.org/devdocs/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html
"""

import numpy as np
import pandas as pd


def compute_weights(n, alpha):

    weights = pd.DataFrame()

    for _n in range(1, n):

        w = [(1.0 - alpha) ** (_n - t) for t in range(_n + 1)]
        w = pd.Series(w, name=str(_n) + "_start")
        weights = pd.concat([weights, w], axis=1)

    return weights


# weights = compute_weights(n=N, alpha=0.1)


def generate_data_ewm(
    alpha,
    mu=None,
    sigma=None,
    nt=None,
    num=None,
    returns=None,
):
    """Generate synthetic data for Exponential Weighted Average test.
    The inputs (x) and their associated outputs (y) are created.

    Parameters
    ----------
    alpha : float, between 0.0 and 1.0
        Closer to 1.0 values imply that the next value dominates the next outcome,
        closer to 0.0 values imply that the previous ewm value dominates the next outcome.
    mu : float
        mean
    sigma : float
        standard deviation
    nt : int
        number of timestamps
    num : int
        number of timeseries
    returns : pandas.DataFrame
        Existing returns, by default None

    Returns
    -------
    returns : pandas.DataFrame
        Normally distributed returns.
    ewm : pandas.DataFrame
        Associated exponentially weighted average.

    Notes
    -----
    - NO shift is applied at this level.
    - Parameter `adjust=True` for pandas.DataFrame.ewm function.
    """

    if returns is None:
        returns = np.random.normal(loc=mu, scale=sigma, size=(nt, num))
        returns = pd.DataFrame(returns)

    ewm = returns.ewm(alpha=alpha, adjust=True)
    ewm = ewm.mean()

    return returns, ewm


if __name__ == "__main__":
    """Small experimental setup to explore the effects of alpha.

    For alpha=0.9, the EWM series has a stronger memory, and thus the current value
    dominates the next EWM value.

    Instead, for alpha=0.1, the next EWM series has a stronger inertia,
    and thus the previous EWM value dominates the next EWM value.

    In the variables check_*,
    - left col. are the returns;
    - right col. is the EWM values.
    """

    MU = 0.001
    SIGMA = 0.1
    NT = 100

    # -------------------------------------------------------------------------
    x, y = generate_data_ewm(alpha=0.9, mu=MU, sigma=SIGMA, nt=NT, num=1)
    check_memory = pd.concat([x, y], axis=1).copy(deep=True)

    # -------------------------------------------------------------------------
    x, y = generate_data_ewm(alpha=0.1, returns=x)
    check_inertia = pd.concat([x, y], axis=1).copy(deep=True)

    # breakpoint()
