"""PyTorch interface of cuPIQP: ``cupiqp.torch``.

Importing this module imports PyTorch. ``import cupiqp`` alone does not; the
attribute ``cupiqp.torch`` loads this module on first access. See
:mod:`cupiqp.interfaces.torch` for the implementation.
"""

from .interfaces.torch import *  # noqa: F401,F403
from .interfaces.torch import __all__  # noqa: F401
