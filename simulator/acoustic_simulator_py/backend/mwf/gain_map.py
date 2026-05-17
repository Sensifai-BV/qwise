"""Neural-guided time-frequency gain-map placeholder.

Port of ``mwf/get_tf_gain_map.m``. Returns all-ones until the
Transformer/Mamba checkpoint is wired in — same behaviour as the MATLAB
stub, so downstream code keeps working unchanged.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# ``cfg`` is unused for now but kept on the signature so future neural
# back-ends can reach the config without changing every call site.
def get_tf_gain_map(X: ArrayLike, cfg=None) -> NDArray[np.float64]:  # noqa: D401
    """Return ``[nbin]`` per-bin gain (currently all ones)."""
    Xa = np.asarray(X)
    nbin = Xa.shape[0]
    return np.ones(nbin, dtype=np.float64)


__all__ = ["get_tf_gain_map"]
