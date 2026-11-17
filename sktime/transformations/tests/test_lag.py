"""Tests for Lag transformer."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).

__author__ = ["fkiraly"]

import itertools

import numpy as np
import pandas as pd
import pytest

from sktime.datatypes import get_examples
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.lag import Lag
from sktime.utils._testing.series import _make_series

# some examples with range vs time index, univariate vs multivariate (mv)
X_range_idx = get_examples("pd.DataFrame")[0]
X_range_idx_mv = get_examples("pd.DataFrame")[1]
X_time_idx = _make_series()
X_time_idx_mv = _make_series(n_columns=2)

# all fixtures
X_fixtures = [X_range_idx, X_range_idx_mv, X_time_idx, X_time_idx_mv]

# fixtures with time index
X_time_fixtures = [X_time_idx, X_time_idx_mv]

index_outs = ["original", "extend", "shift"]


@pytest.mark.skipif(
    not run_test_for_class(Lag),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("X", X_fixtures)
@pytest.mark.parametrize("index_out", index_outs)
def test_lag_fit_transform_out_index(X, index_out):
    """Test that index sets of fit_transform output behave as expected."""
    t = Lag(2, index_out=index_out)
    Xt = t.fit_transform(X)

    if index_out == "original":
        assert Xt.index.equals(X.index)
    elif index_out == "extend":
        assert X.index.isin(Xt.index).all()
        assert len(Xt) == len(X) + 2
    elif index_out == "shift":
        assert len(Xt) == len(X)
        assert X.index[2:].isin(Xt.index).all()
        assert not X.index[:2].isin(Xt.index).any()


@pytest.mark.skipif(
    not run_test_for_class(Lag),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("X", X_fixtures)
@pytest.mark.parametrize("index_out", index_outs)
@pytest.mark.parametrize("remember_data", [True, False])
def test_lag_fit_transform_out_values(X, index_out, remember_data):
    """Test that index sets of fit_transform output behave as expected."""
    t = Lag(2, index_out=index_out, remember_data=remember_data)
    if isinstance(X, pd.Series):
        X = pd.DataFrame(X)
    X_fit = X[:2]
    X_trafo = X[2:]
    Xt = t.fit(X_fit).transform(X_trafo)

    if index_out in ["original", "extend"]:
        if remember_data:
            np.testing.assert_equal(Xt.iloc[0].values, X_fit.iloc[0].values)
        else:  # remember_data == False
            assert all(Xt.iloc[0].isna().values)
        if len(Xt) > 2:
            np.testing.assert_equal(Xt.iloc[2].values, X_trafo.iloc[0].values)

    elif index_out == "shift":
        np.testing.assert_equal(Xt.values, X_trafo.values)


@pytest.mark.skipif(
    not run_test_for_class(Lag),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("X", X_fixtures)
@pytest.mark.parametrize("index_out", index_outs)
@pytest.mark.parametrize("lag", [2, [2, 4], [-1, 0, 5]])
def test_lag_fit_transform_columns(X, index_out, lag):
    """Test that columns of fit_transform output behave as expected."""
    t = Lag(lags=lag, index_out=index_out)
    Xt = t.fit_transform(X)

    if isinstance(lag, list):
        len_lag = len(lag)
    else:
        len_lag = 1

    def ncols(obj):
        if isinstance(obj, pd.DataFrame):
            return len(obj.columns)
        else:
            return 1

    assert ncols(Xt) == ncols(X) * len_lag


@pytest.mark.skipif(
    not run_test_for_class(Lag),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("X", X_fixtures)
@pytest.mark.parametrize("index_out", index_outs)
@pytest.mark.parametrize("lags", [2, [2, 4]])
def test_lag_fit_transform_column_names(X, index_out, lags):
    """Test expected column names."""
    t = Lag(lags=lags, index_out=index_out)
    Xt = t.fit_transform(X)

    if isinstance(Xt, pd.DataFrame):
        lag_col_names = set(Xt.columns)

        if isinstance(X, pd.DataFrame):
            col_names = X.columns
        elif isinstance(X, pd.Series):
            col_names = [X.name if X.name else 0]
        else:
            pass

        lags = [lags] if isinstance(lags, int) else lags
        expected = {
            f"lag_{lag}__{col}" for lag, col in itertools.product(lags, col_names)
        }
        assert lag_col_names == expected

    elif isinstance(Xt, pd.Series):
        assert Xt.name is None


@pytest.mark.skipif(
    not run_test_for_class(Lag),
    reason="Run if tests if module has changed.",
)
@pytest.mark.parametrize(
    "index",
    [pd.RangeIndex(0, 12, 2), pd.Index([10, 15, 20, 25, 30, 35])],
)
@pytest.mark.parametrize("index_out", ["original", "extend", "shift"])
def test_lag_integer_index_with_step(index, index_out):
    """Lag on an evenly spaced integer index with step other than 1.

    A lag of 1 must move each value to the next time point, as for a
    DatetimeIndex with the same spacing, not to the label plus 1.
    """
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=index)
    step = index[1] - index[0]

    Xt = Lag(lags=[0, 1], index_out=index_out).fit_transform(X)

    lag1 = Xt["lag_1__a"]
    # every value appears again one time point later, never at label + 1
    for label, value in zip(index, X["a"]):
        if label + step in lag1.index:
            assert lag1.loc[label + step] == value
    assert not any((label + 1) in lag1.index for label in index if step != 1)
    if index_out == "original":
        assert Xt.index.equals(index)
        np.testing.assert_array_equal(lag1.to_numpy()[1:], X["a"].to_numpy()[:-1])
        assert np.isnan(lag1.iloc[0])


@pytest.mark.skipif(
    not run_test_for_class(Lag),
    reason="Run if tests if module has changed.",
)
def test_lag_integer_index_with_step_remembers_data():
    """A single new time point is lagged with the step of the data seen in fit."""
    index = pd.RangeIndex(0, 12, 2)
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=index)

    trafo = Lag(lags=1, index_out="original").fit(X.iloc[:5])
    Xt = trafo.transform(X.iloc[5:])

    assert list(Xt.index) == [10]
    assert Xt.iloc[0, 0] == 5.0
