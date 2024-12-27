import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def add_pd(doctest_namespace):
    doctest_namespace["pd"] = pd
