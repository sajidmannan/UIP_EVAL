from __future__ import annotations

from matsciml.lightning.callbacks import (
    ExponentialMovingAverageCallback,
    ManualGradientClip,
)
from pytorch_lightning.callbacks import StochasticWeightAveraging

"""
This script is not intended to work entirely, but
shows the key elements needed for training (for
the Intel folks, "BKM").

These are:

- For energies, use ``AtomWeightedMSE`` as the metric. The new default behavior
  for ``ForceRegressionTask`` uses this, but if you use another task to represent
  it this may need to be manually set.
- Use stochastic weight averaging callback
- Use exponential moving average callback
- Use gradient clipping

An important thing to note about EMA is that the exponential weights
are used _outside_ of training: for validation/testing/ASE usage,
the EMA weights will be used instead of the "vanilla" model. This
means that logged values are not necessarily directly comparable
between training and validation, or rather, validation should be
outperforming training.
"""

# pass into trainer configuration
callbacks = [
    StochasticWeightAveraging(swa_lrs=1e-2, swa_epoch_start=1),
    ExponentialMovingAverageCallback(decay=0.99),
    ManualGradientClip(value=10.0, algorithm="norm"),
]
