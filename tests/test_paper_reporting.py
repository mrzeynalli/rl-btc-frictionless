"""Reporting-only checks; these do not train or alter the selected policies."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from paper_tables import returns


def test_equity_returns_include_initial_capital():
    equity = pd.Series([11000.0, 9900.0, 11880.0])
    np.testing.assert_allclose(returns(equity), [0.1, -0.1, 0.2])
