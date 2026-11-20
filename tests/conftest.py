import pandas as pd
import pytest

from carbon_forecast.data import clean_series
from carbon_forecast.model import CarbonForecaster
from carbon_forecast.synthetic import synthetic_intensity


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    return synthetic_intensity(days=60, seed=1)


@pytest.fixture(scope="session")
def series(raw) -> pd.Series:
    return clean_series(raw)


@pytest.fixture(scope="session")
def fitted_model(series) -> CarbonForecaster:
    return CarbonForecaster(origin_stride=4).fit(series)
