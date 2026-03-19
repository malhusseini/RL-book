"""strategies — Three causal daily return streams.

  conservative  PCA residual mean reversion (Avellaneda-Lee).
  medium        Pairs trading via rolling correlation and spread z-score.
  aggressive    Variance-risk-premium proxy (short-vol profile).
"""

from .conservative import build_conservative
from .medium import build_medium
from .aggressive import build_aggressive
from .build import build_strategy_series

__all__ = [
    "build_conservative",
    "build_medium",
    "build_aggressive",
    "build_strategy_series",
]
